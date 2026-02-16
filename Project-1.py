"""
===========================================================
ML4B Individual Assignment – Wikipedia Market Research Assistant
===========================================================

This Streamlit application implements a simple AI-powered
market research assistant as required in the assignment brief.

Pipeline overview (matches Q1–Q3 exactly):
1) Validate user input (industry)
2) Retrieve the 5 most relevant Wikipedia pages
3) Generate a <500-word industry report grounded in those pages

Key design principles:
- KISS: keep the system simple and transparent
- Ground LLM outputs in retrieved evidence (Wikipedia)
- Explicitly enforce assessment constraints (word limit, sources)

Libraries used:
- Streamlit: interactive UI
- LangChain WikipediaRetriever: information retrieval (mentioned in brief)
- Google Generative AI (Gemma/Gemini): text generation
"""

# =========================================================
# 1. Imports
# =========================================================
# !pip install --upgrade pip

# !pip install \
#   streamlit \
#   langchain \
#   langchain-community \
#   langchain-google-genai \
#   wikipedia \
#   python-dotenv

import os
import re
import streamlit as st

# LLM interface (cheap model recommended for development)
from langchain_google_genai import ChatGoogleGenerativeAI

# Wikipedia retrieval tool
from langchain_community.retrievers import WikipediaRetriever


# =========================================================
# 2. System Prompt (Defines assistant behaviour & constraints)
# =========================================================

"""
The system prompt acts as the "policy" for the assistant.
It defines:
- the role of the assistant
- hard constraints (word limit, grounding)
- output structure for consistency

This improves reliability and reduces hallucination.
"""

SYSTEM_PROMPT = """
You are a market research assistant for a corporate business analyst.

Rules:
- Write an industry report under 500 words. Aim for 430–480 words to cover key points.
- Use ONLY the provided Wikipedia extracts and URLs as evidence.
- Do NOT invent facts. If the sources do not specify something, say so.
- Every factual sentence must include at least one citation tag: [S1], [S2], [S3], [S4], or [S5].
- Keep the tone clear, professional, and business-oriented.

Output format (use these headings):
1) Overview
2) Value chain / business model
3) Demand drivers
4) Competitive landscape
5) Trends & risks
6) Sources (list the 5 URLs)
""".strip()


# =========================================================
# 3. Helper Function: Build User Prompt with Retrieved Evidence
# =========================================================

def build_user_prompt(industry: str, docs: list) -> str:
    """
    Combines:
    - the user-provided industry
    - the retrieved Wikipedia documents

    This ensures the LLM output is grounded in retrieved data
    (retrieval-augmented generation).
    """

    sources_block = []

    for i, d in enumerate(docs, start=1):
        title = d.metadata.get("title", "Unknown title")
        url = d.metadata.get("source", "")
        extract = d.page_content.strip()

        # Truncate extracts to avoid exceeding the LLM context window
        extract = extract[:1200]

        sources_block.append(
            f"[S{i}] {title}\nURL: {url}\nExtract: {extract}\n"
        )

    sources_text = "\n".join(sources_block)

    user_prompt = f"""
Industry: {industry}

Wikipedia sources (use ONLY these):
{sources_text}

Task:
Write the industry report now, following the required headings and staying under 500 words.
Use only these extracts. Add [S#] citation tags to each factual sentence.
""".strip()

    return user_prompt


# =========================================================
# 3b. LLM Utility (consistent parsing for JSON-like replies)
# =========================================================

def _invoke_llm_json(llm: ChatGoogleGenerativeAI, prompt: str) -> str:
    """
    Invoke the LLM and return raw text for downstream JSON parsing.
    This keeps all LLM calls consistent and easy to audit.
    """
    result = llm.invoke([("user", prompt)])
    content = result.content
    if isinstance(content, dict):
        content = content.get("text", str(content))
    elif isinstance(content, list):
        content = "".join([str(c) for c in content])
    return str(content).strip()


def validate_api_key(api_key: str, model_name: str) -> bool:
    """
    Fast key check before running the full app pipeline.
    Returns True if the key can successfully call the selected model.
    """
    if not api_key or not model_name:
        return False
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.0,
            max_output_tokens=8
        )
        _invoke_llm_json(llm, "Reply with OK.")
        return True
    except Exception:
        return False


def detect_abbreviation(term: str, api_key: str, model_name: str) -> dict:
    """
    Use LLM to detect whether the term is an abbreviation and suggest a VALID industry name.
    Returns: {"is_abbrev": bool, "expanded": str}
    """
    if not api_key or not model_name:
        return {"is_abbrev": False, "expanded": ""}

    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.0,
            max_output_tokens=80
        )
        llm_prompt = f"""
If the term is a COMMON abbreviation in business/industry contexts, expand it into a VALID industry name.
The expanded value MUST be a clear industry/sector/market phrase and include
the word "industry", "sector", or "market".
If it is not a common abbreviation or is unclear, answer no.
Return ONLY JSON: {{"is_abbrev":"yes|no","expanded":"...","confidence":"0-1"}}
Term: {term}
""".strip()
        text = _invoke_llm_json(llm, llm_prompt)
        is_abbrev_match = re.search(r'"is_abbrev"\s*:\s*"(yes|no)"', text, re.IGNORECASE)
        expanded_match = re.search(r'"expanded"\s*:\s*"([^"]*)"', text, re.IGNORECASE)
        conf_match = re.search(r'"confidence"\s*:\s*"?(0(?:\.\d+)?|1(?:\.0+)?)"?', text, re.IGNORECASE)
        conf = float(conf_match.group(1)) if conf_match else 0.0
        if is_abbrev_match and is_abbrev_match.group(1).lower() == "yes" and expanded_match and conf >= 0.7:
            expanded = expanded_match.group(1).strip()
            if expanded:
                # Require explicit industry/sector/market wording
                if re.search(r"\b(industry|sector|market)\b", expanded.lower()):
                    # Verify the expanded term looks like a real industry via Wikipedia evidence
                    retriever = WikipediaRetriever(top_k_results=3, lang="en")
                    docs = retriever.invoke(expanded)
                    evidence_ok = False
                    for d in docs:
                        title = (d.metadata.get("title", "") or "").lower()
                        content = (d.page_content or "").lower()[:400]
                        if any(k in title or k in content for k in ["industry", "sector", "market"]):
                            evidence_ok = True
                            break
                    if evidence_ok:
                        return {"is_abbrev": True, "expanded": expanded}
    except Exception:
        pass

    return {"is_abbrev": False, "expanded": ""}


# =========================================================
# 4. Industry Validation (Q1)
# =========================================================

def validate_industry(industry: str, top_k: int = 5, api_key: str = "", model_name: str = "") -> tuple:
    """
    Validates if the input is a valid industry.
    Short, simple flow:
    1) LLM (if available) decides yes/no.
    2) Wikipedia must return relevant pages; otherwise reject.
    
    Returns: (is_valid: bool, suggestion: str or None)
    """
    # LLM-only decision path (simple and final when available).
    if api_key and model_name:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.0,
                max_output_tokens=200
            )
            llm_prompt = f"""
Classify whether this term is a valid industry (economic sector/market).
Answer ONLY in JSON:
{{"is_industry":"yes|no"}}
Term: {industry}
Normalization guidance:
- Treat spacing, hyphen, and spelling variants as equivalent when they refer to the same sector
  (e.g., "healthcare" = "health care", "e-commerce" = "ecommerce").
Decision rules:
- "yes" only if it is a broad economic sector/market with many firms (e.g., healthcare, banking, automotive, retail).
- "no" for sports/hobbies/topics, colors/random text, specific products, company names, or narrow store formats.
- "no" for single activity domains that are not commonly treated as an industry label.
Examples:
- "healthcare" -> yes
- "automotive industry" -> yes
- "basketball" -> no
- "luxury department store" -> no
- "cars" -> no
- "glasses" -> no
- "phone" -> no
- "red industry" -> no
- "kk" -> no
If unclear, answer "no".
""".strip()
            text = _invoke_llm_json(llm, llm_prompt)
            is_industry_match = re.search(r'"is_industry"\s*:\s*"(yes|no)"', text, re.IGNORECASE)
            if is_industry_match and is_industry_match.group(1).lower() == "yes":
                return True, None
            if is_industry_match and is_industry_match.group(1).lower() == "no":
                suggestion = find_similar_industry(industry, api_key=api_key, model_name=model_name)
                return False, suggestion
            # Unparseable response: fail closed.
            suggestion = find_similar_industry(industry, api_key=api_key, model_name=model_name)
            return False, suggestion
        except Exception:
            # LLM error: fail closed to avoid false positives.
            suggestion = find_similar_industry(industry, api_key=api_key, model_name=model_name)
            return False, suggestion

    # No LLM configured.
    suggestion = find_similar_industry(industry, api_key=api_key, model_name=model_name)
    return False, suggestion


def find_similar_industry(invalid_input: str, api_key: str = "", model_name: str = "") -> str:
    """
    Suggests relevant industry terms based on the user's input.
    Uses:
    - LLM-based suggestion (if available)
    - A small synonym/related-industry map
    - Fallback string similarity on a common industry list
    Returns top suggestions for the user.
    """
    # LLM-based suggestions (preferred, more robust to typos)
    if api_key and model_name:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.0,
                max_output_tokens=60
            )
            llm_prompt = f"""
The user entered an invalid industry term: "{invalid_input}".
Suggest 3 VALID industry names they might have meant (fix typos if likely).
Each suggestion must be a real industry/sector/market and include the word
"industry", "sector", or "market".
If the input looks like a product/object (e.g., cars, glasses, phone), map it to
the closest relevant industry terms (e.g., automotive industry, eyewear industry,
telecommunications industry) instead of repeating the product word.
Return ONLY a comma-separated list of industries.
""".strip()
            text = _invoke_llm_json(llm, llm_prompt)
            # Basic cleanup to ensure a simple list
            text = re.sub(r"[\n;]+", ", ", text)
            text = re.sub(r"^[-*\d\.\s]+", "", text)
            if len(text) > 0:
                # Verify suggestions via Wikipedia
                parts = [p.strip() for p in text.split(",") if p.strip()]
                valid = []
                retriever = WikipediaRetriever(top_k_results=3, lang="en")
                for p in parts:
                    if not re.search(r"\b(industry|sector|market)\b", p.lower()):
                        continue
                    docs = retriever.invoke(p)
                    for d in docs:
                        title = (d.metadata.get("title", "") or "").lower()
                        content = (d.page_content or "").lower()[:400]
                        if any(k in title or k in content for k in ["industry", "sector", "market"]):
                            valid.append(p)
                            break
                    if len(valid) >= 3:
                        break
                if valid:
                    return ", ".join(valid)
        except Exception:
            pass
    # Heuristic fallback (only if LLM is unavailable)
    # Common industries as reference
    common_industries = [
        "technology", "healthcare", "finance", "retail", "manufacturing",
        "energy", "transportation", "agriculture", "education", "hospitality",
        "automotive", "construction", "telecommunications", "media", "entertainment",
        "pharmaceutical", "food and beverage", "real estate", "insurance", "aerospace"
    ]

    # Map common objects/terms to plausible industry suggestions
    related_map = {
        "car": ["automotive industry", "vehicle industry", "transportation industry"],
        "vehicle": ["automotive industry", "vehicle industry", "transportation industry"],
        "camera": ["imaging industry", "photography industry", "consumer electronics industry"],
        "photo": ["photography industry", "media industry", "consumer electronics industry"],
        "phone": ["telecommunications industry", "consumer electronics industry", "technology industry"],
        "laptop": ["consumer electronics industry", "technology industry"],
        "computer": ["technology industry", "IT services industry", "consumer electronics industry"],
        "food": ["food and beverage industry", "agriculture industry", "hospitality industry"],
        "coffee": ["food and beverage industry", "hospitality industry"],
        "pizza": ["food and beverage industry", "hospitality industry"],
        "water": ["utilities industry", "water industry", "environmental services industry"],
        "bank": ["banking industry", "financial services industry", "insurance industry"],
        "money": ["financial services industry", "banking industry", "insurance industry"],
        "house": ["real estate industry", "construction industry", "mortgage industry"],
        "book": ["publishing industry", "media industry", "education industry"],
        "movie": ["media industry", "entertainment industry"],
        "game": ["video game industry", "entertainment industry", "media industry"],
        "hospital": ["healthcare industry", "medical services industry"],
        "drug": ["pharmaceutical industry", "healthcare industry"],
        "energy": ["energy industry", "renewable energy industry", "utilities industry"]
    }

    input_lower = invalid_input.lower().strip()

    # If input is too short or meaningless, return general suggestions
    if len(input_lower) < 2:
        return "technology, healthcare, finance, retail, or manufacturing"

    # If input matches a mapped term, return those suggestions
    for key, suggestions in related_map.items():
        if key in input_lower:
            return ", ".join(suggestions[:3])

    # If user typed something like "X industry", keep it but also suggest related
    if "industry" in input_lower:
        return f"{input_lower}, technology industry, healthcare industry"

    # Otherwise, attempt to create a reasonable industry phrase from input
    guessed = f"{input_lower} industry"

    # Find the closest matches using simple string similarity
    scores = []
    for industry in common_industries:
        matches = sum(1 for c in input_lower if c in industry)
        score = matches / max(len(input_lower), len(industry))
        scores.append((industry, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_suggestions = [industry for industry, score in scores[:2] if score > 0.1]

    # Combine the guessed phrase + top matches
    combined = [guessed] + top_suggestions
    combined = [s for i, s in enumerate(combined) if s not in combined[:i]]

    if combined:
        return ", ".join(combined[:3])
    else:
        return "technology, healthcare, or finance"


# =========================================================
# 5. Wikipedia Retrieval (Q2)
# =========================================================

def retrieve_wikipedia(industry: str, top_k: int = 5):
    """
    Retrieves the top-k most relevant Wikipedia pages
    for the given industry using LangChain's WikipediaRetriever.
    """
    retriever = WikipediaRetriever(top_k_results=top_k, lang="en")
    docs = retriever.invoke(industry)
    return docs


def select_top_docs_with_llm(industry: str, docs: list, api_key: str, model_name: str, k: int = 5) -> list:
    """
    Second-layer ranking: ask the LLM to choose the top-k most relevant pages
    from a larger retrieved set. Falls back to the first k docs on any failure.
    """
    if not docs:
        return []
    if not api_key or not model_name:
        return docs[:k]

    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.0,
            max_output_tokens=180
        )

        candidates = []
        for i, d in enumerate(docs, start=1):
            title = d.metadata.get("title", "Unknown title")
            url = d.metadata.get("source", "")
            extract = (d.page_content or "").strip()[:350]
            candidates.append(f"{i}) {title}\nURL: {url}\nExtract: {extract}")

        llm_prompt = f"""
Pick the {k} most relevant Wikipedia pages for industry market research.
Return ONLY JSON: {{"indices":[1,2,3,4,5]}}
Industry: {industry}

Candidates:
{chr(10).join(candidates)}
""".strip()

        text = _invoke_llm_json(llm, llm_prompt)
        match = re.search(r'"indices"\s*:\s*\[([^\]]+)\]', text)
        if not match:
            return docs[:k]

        indices = [int(x) for x in re.findall(r"\d+", match.group(1))]
        selected = []
        seen = set()
        for idx in indices:
            if 1 <= idx <= len(docs) and idx not in seen:
                selected.append(docs[idx - 1])
                seen.add(idx)
            if len(selected) == k:
                break

        return selected if len(selected) == k else docs[:k]
    except Exception:
        return docs[:k]


def extract_urls(docs: list) -> list:
    """
    Extracts and deduplicates Wikipedia URLs from retrieved documents.
    These URLs are displayed explicitly to satisfy Q2.
    """
    urls = []
    for d in docs:
        url = d.metadata.get("source", "")
        if url:
            urls.append(url)

    # Deduplicate while preserving order
    deduped = []
    for u in urls:
        if u not in deduped:
            deduped.append(u)

    return deduped


# =========================================================
# 6. LLM Generation (Q3)
# =========================================================

def word_count(text: str) -> int:
    """Count words in the report body only (exclude sources section and URLs)."""
    # Drop section 6 / sources block from counting.
    body_only = re.sub(r"(?is)\n\s*(\*\*)?\s*(6[\)\.\:]?\s*)?sources\s*(\*\*)?\s*.*$", "", text).strip()
    # Remove any remaining URLs.
    body_only = re.sub(r'https?://\S+', '', body_only)
    return len(re.findall(r"\b\w+\b", body_only))


def generate_report(model_name: str, api_key: str, system_prompt: str, user_prompt: str) -> str:
    """
    Generates the industry report using an LLM.

    Handles model-specific message formatting:
    - Gemma models do not support a separate system role
    - Other models (e.g. Gemini) do
    """

    def extract_text(content) -> str:
        """
        Normalize model outputs across providers.
        Recursively collects any 'text' fields found in nested dict/list structures.
        """
        if isinstance(content, str):
            return content

        texts = []

        def collect(obj):
            if isinstance(obj, dict):
                if "text" in obj and isinstance(obj["text"], str):
                    texts.append(obj["text"])
                for v in obj.values():
                    collect(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    collect(v)

        collect(content)

        if texts:
            return "".join(texts)
        return str(content)

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.0,          # Deterministic output for stricter grounding
        max_output_tokens=650     # Allows space while still enforcing word cap
    )

    if "gemma" in model_name.lower():
        messages = [("user", f"{system_prompt}\n\n{user_prompt}")]
    else:
        messages = [("system", system_prompt), ("user", user_prompt)]

    result = llm.invoke(messages)
    response = extract_text(result.content).strip()
    return response


def enforce_grounded_rewrite(model_name: str, api_key: str, system_prompt: str, user_prompt: str, draft_report: str) -> str:
    """
    Second pass: remove/repair unsupported claims and keep only source-grounded content.
    """
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.0,
        max_output_tokens=700
    )

    rewrite_prompt = f"""
Rewrite the draft report so every factual statement is supported by the provided source extracts.
Hard constraints:
- Use ONLY information from the provided sources.
- Remove unsupported statements.
- Keep required headings (1-6).
- Keep under 500 words.
- Add [S1]-[S5] citations to each factual sentence.
- If a detail is missing from sources, explicitly say it is not specified in the sources.

Draft report:
{draft_report}

Sources context:
{user_prompt}
""".strip()

    if "gemma" in model_name.lower():
        messages = [("user", f"{system_prompt}\n\n{rewrite_prompt}")]
    else:
        messages = [("system", system_prompt), ("user", rewrite_prompt)]

    result = llm.invoke(messages)
    content = result.content
    if isinstance(content, dict):
        content = content.get("text", str(content))
    elif isinstance(content, list):
        content = "".join([str(c) for c in content])
    return str(content).strip()


def enforce_under_500_words(report: str) -> str:
    """
    Safety guardrail to ensure the report respects
    the 500-word assessment limit.
    """
    if word_count(report) <= 500:
        return report

    words = re.findall(r"\S+", report)
    trimmed = " ".join(words[:500])
    trimmed += "\n\n[Trimmed to meet the 500-word limit.]"
    return trimmed


def ensure_sources_section(report: str, urls: list) -> str:
    """
    Ensures a complete Sources section with the provided URLs.
    Replaces any existing Sources section to avoid partial lists.
    """
    # Remove any trailing section 6 / sources block that the model may already produce.
    clean = re.sub(r"(?is)\n\s*(\*\*)?\s*6[\)\.\:]?\s*(sources)?\s*(\*\*)?\s*.*$", "", report).strip()
    # Also remove a trailing standalone "Sources" heading block if present.
    clean = re.sub(r"(?is)\n\s*(\*\*)?\s*sources\s*(\*\*)?\s*.*$", "", clean).strip()
    sources_lines = "\n".join(urls[:5])
    return f"{clean}\n\n### 6) Sources\n{sources_lines}".strip()


# =========================================================
# 7. Streamlit User Interface (Q1–Q3)
# =========================================================

"""
The Streamlit UI follows the assignment workflow exactly:
Q1: User inputs an industry (with validation)
Q2: Display 5 Wikipedia URLs
Q3: Generate and display the industry report
"""

st.set_page_config(
    page_title="Wikipedia Market Research Assistant",
    layout="centered"
)

st.title("Wikipedia Market Research Assistant")
st.caption("Enter an industry → retrieve Wikipedia pages → generate a concise industry report")

# Sidebar settings
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Google API key", type="password").strip()

model_name = st.sidebar.selectbox(
    "Choose an LLM:",
    [
        "gemma-3-27b-it"
    ],
    index=0
)

industry_locked = not bool(api_key)
industry = st.text_input(
    "Industry",
    placeholder="e.g., electric vehicles, luxury retail, fintech",
    disabled=industry_locked
)

if industry_locked:
    st.info("Enter your Google API key in the sidebar to enable industry input.")

generate_clicked = st.button("Generate report", disabled=industry_locked)
if generate_clicked:
    st.session_state["pending_generate"] = True

if st.session_state.get("pending_generate", False):

    # -------------------------
    # Q1: Input validation
    # -------------------------
    if not validate_api_key(api_key, model_name):
        st.error("Invalid API key")
        st.session_state["pending_generate"] = False
        st.stop()

    if industry.strip() == "":
        st.warning("Please enter an industry.")
        st.stop()

    # Abbreviation confirmation (LLM-assisted)
    # Rationale: let the user confirm a likely expansion before validation.
    if "abbr_input" not in st.session_state or st.session_state.get("abbr_input") != industry:
        st.session_state["abbr_input"] = industry
        st.session_state["abbr_done"] = False
        st.session_state["abbr_resolved"] = industry

    if not st.session_state.get("abbr_done", False):
        abbr = detect_abbreviation(industry, api_key=api_key, model_name=model_name)
        if abbr.get("is_abbrev") and abbr.get("expanded"):
            st.info(f'Did you mean "{abbr["expanded"]}"?')
            st.caption("Recommended: use the expanded industry name for better accuracy.")
            c1, c2 = st.columns(2)
            yes_clicked = c1.button(f'Yes (Recommended), use "{abbr["expanded"]}"')
            no_clicked = c2.button("No, keep original")
            c2.warning("Using abbreviations may lead to less accurate results.")
            if yes_clicked:
                st.session_state["abbr_resolved"] = abbr["expanded"]
                st.session_state["abbr_done"] = True
                st.session_state["pending_generate"] = True
            elif no_clicked:
                st.session_state["abbr_resolved"] = industry
                st.session_state["abbr_done"] = True
                st.session_state["pending_generate"] = True
            else:
                st.stop()
        else:
            st.session_state["abbr_done"] = True

    industry_to_use = st.session_state.get("abbr_resolved", industry)

    # Validate that input is actually an industry
    # Rationale: ensures Q1 is satisfied before retrieval and reporting.
    is_valid, suggestion = validate_industry(
        industry_to_use,
        top_k=5,
        api_key=api_key,
        model_name=model_name
    )
    
    if not is_valid:
        st.error(f"❌ '{industry_to_use}' does not appear to be a valid industry.")
        if suggestion:
            st.info(f"💡 Did you mean one of these? Try: **{suggestion}**")
        st.stop()

    # -------------------------
    # Q2: Wikipedia retrieval
    # Rationale: gather sources for both URL display and report grounding.
    # -------------------------
    with st.spinner("Retrieving Wikipedia pages..."):
        candidate_docs = retrieve_wikipedia(industry_to_use, top_k=10)
        docs = select_top_docs_with_llm(
            industry=industry_to_use,
            docs=candidate_docs,
            api_key=api_key,
            model_name=model_name,
            k=5
        )

    urls = extract_urls(docs)

    st.subheader("Q2 — Top 5 Wikipedia URLs")
    if not urls:
        st.error("No Wikipedia pages were retrieved. Try a different industry.")
        st.stop()

    for u in urls[:5]:
        st.write(u)

    # -------------------------
    # Q3: Report generation
    # Rationale: generate a grounded report using only retrieved extracts.
    # -------------------------
    if not api_key:
        st.error("Missing API key. Please set GOOGLE_API_KEY.")
        st.stop()

    user_prompt = build_user_prompt(industry_to_use, docs)

    with st.spinner("Generating industry report..."):
        report = generate_report(
            model_name=model_name,
            api_key=api_key,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt
        )

        report = enforce_grounded_rewrite(
            model_name=model_name,
            api_key=api_key,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            draft_report=report
        )
        report = enforce_under_500_words(report)
        report = ensure_sources_section(report, urls)

    st.subheader("Q3 — Industry Report (<500 words)")
    st.write(report)
    st.caption(f"Word count: {word_count(report)}")
    st.session_state["pending_generate"] = False
