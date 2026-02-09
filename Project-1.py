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
- Write an industry report under 500 words.
- Use ONLY the provided Wikipedia extracts and URLs as evidence.
- Do NOT invent facts. If the sources do not specify something, say so.
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
            f"{i}) {title}\nURL: {url}\nExtract: {extract}\n"
        )

    sources_text = "\n".join(sources_block)

    user_prompt = f"""
Industry: {industry}

Wikipedia sources (use ONLY these):
{sources_text}

Task:
Write the industry report now, following the required headings and staying under 500 words.
""".strip()

    return user_prompt


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
    def normalize_tokens(text: str) -> list:
        return re.findall(r"[a-zA-Z]+", text.lower())

    def is_disambiguation(title: str, content: str) -> bool:
        title_l = title.lower()
        content_l = content.lower()
        return "(disambiguation)" in title_l or "may refer to" in content_l[:400]

    input_lower = industry.lower().strip()
    tokens = normalize_tokens(input_lower)

    # Simple guardrail for color-only inputs
    if tokens in (["red"], ["blue"], ["green"], ["yellow"], ["purple"], ["orange"],
                  ["black"], ["white"], ["brown"], ["pink"], ["gray"], ["grey"],
                  ["cyan"], ["magenta"], ["gold"], ["silver"]) or \
       tokens in (["red", "industry"], ["blue", "industry"], ["green", "industry"],
                  ["yellow", "industry"], ["purple", "industry"], ["orange", "industry"],
                  ["black", "industry"], ["white", "industry"], ["brown", "industry"],
                  ["pink", "industry"], ["gray", "industry"], ["grey", "industry"],
                  ["cyan", "industry"], ["magenta", "industry"], ["gold", "industry"],
                  ["silver", "industry"]):
        suggestion = find_similar_industry(industry, api_key=api_key, model_name=model_name)
        return False, suggestion

    # 1) LLM decision first (if available)
    if api_key and model_name:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.0,
                max_output_tokens=200
            )
            llm_prompt = f"""
Is this a valid INDUSTRY? Answer ONLY in JSON:
{{"is_industry": "yes" or "no", "reason": "short reason"}}
Term: {industry}
If unclear or not an economic sector/market, answer "no".
""".strip()
            result = llm.invoke([("user", llm_prompt)])
            content = result.content
            if isinstance(content, dict):
                content = content.get("text", str(content))
            elif isinstance(content, list):
                content = "".join([str(c) for c in content])
            text = content.strip()
            is_industry_match = re.search(r'"is_industry"\\s*:\\s*"(yes|no)"', text, re.IGNORECASE)
            if is_industry_match and is_industry_match.group(1).lower() == "no":
                suggestion = find_similar_industry(industry, api_key=api_key, model_name=model_name)
                return False, suggestion
        except Exception:
            pass

    # 2) Wikipedia must return relevant pages
    retriever = WikipediaRetriever(top_k_results=top_k, lang="en")
    docs = retriever.invoke(industry)
    docs = [d for d in docs if not is_disambiguation(d.metadata.get("title", ""), d.page_content or "")]
    if not docs:
        suggestion = find_similar_industry(industry, api_key=api_key, model_name=model_name)
        return False, suggestion

    # Require some sign of "industry/sector/market" in top results
    evidence_ok = False
    for d in docs[:5]:
        title = (d.metadata.get("title", "") or "").lower()
        content = (d.page_content or "").lower()[:400]
        if "industry" in title or "sector" in title or "market" in title:
            evidence_ok = True
            break
        if "industry" in content or "sector" in content or "market" in content:
            evidence_ok = True
            break

    if not evidence_ok:
        suggestion = find_similar_industry(industry, api_key=api_key, model_name=model_name)
        return False, suggestion

    return True, None


def find_similar_industry(invalid_input: str, api_key: str = "", model_name: str = "") -> str:
    """
    Suggests relevant industry terms based on the user's input.
    Uses:
    - LLM-based suggestion (if available)
    - A small synonym/related-industry map
    - Fallback string similarity on a common industry list
    Returns top suggestions for the user.
    """
    if api_key and model_name:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.2,
                max_output_tokens=60
            )
            llm_prompt = f"""
The user entered an invalid industry term: "{invalid_input}".
Suggest 3 valid industries they might have meant (fix typos if likely).
Return ONLY a comma-separated list of industries.
""".strip()
            result = llm.invoke([("user", llm_prompt)])
            content = result.content
            if isinstance(content, dict):
                content = content.get("text", str(content))
            elif isinstance(content, list):
                content = "".join([str(c) for c in content])
            text = content.strip()
            # Basic cleanup to ensure a simple list
            text = re.sub(r"[\n;]+", ", ", text)
            text = re.sub(r"^[-*\d\.\s]+", "", text)
            if len(text) > 0:
                return text
        except Exception:
            pass
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
    """Utility function to count words in report text (excludes URLs)."""
    # Remove URLs from text before counting
    text_without_urls = re.sub(r'https?://\S+', '', text)
    return len(re.findall(r"\b\w+\b", text_without_urls))


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
        temperature=0.3,          # Lower temperature for factual consistency
        max_output_tokens=650     # Allows space while still enforcing word cap
    )

    if "gemma" in model_name.lower():
        messages = [("user", f"{system_prompt}\n\n{user_prompt}")]
    else:
        messages = [("system", system_prompt), ("user", user_prompt)]

    result = llm.invoke(messages)
    response = extract_text(result.content).strip()
    return response


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
    clean = re.sub(r"(?is)\bSources\b.*$", "", report).strip()
    sources_lines = "\n".join(urls[:5])
    return f"{clean}\n\nSources\n{sources_lines}".strip()


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
        "gemma-3-4b-it",
        "gemma-3-27b-it",
        "gemini-3-flash-preview"
    ],
    index=0
)

industry = st.text_input(
    "Industry",
    placeholder="e.g., electric vehicles, luxury retail, fintech"
)

if st.button("Generate report"):

    # -------------------------
    # Q1: Input validation
    # -------------------------
    if industry.strip() == "":
        st.warning("Please enter an industry.")
        st.stop()

    # Validate that input is actually an industry
    is_valid, suggestion = validate_industry(
        industry,
        top_k=5,
        api_key=api_key,
        model_name=model_name
    )
    
    if not is_valid:
        st.error(f"❌ '{industry}' does not appear to be a valid industry.")
        if suggestion:
            st.info(f"💡 Did you mean one of these? Try: **{suggestion}**")
        st.stop()

    # -------------------------
    # Q2: Wikipedia retrieval
    # -------------------------
    with st.spinner("Retrieving Wikipedia pages..."):
        docs = retrieve_wikipedia(industry, top_k=5)

    urls = extract_urls(docs)

    st.subheader("Q2 — Top 5 Wikipedia URLs")
    if not urls:
        st.error("No Wikipedia pages were retrieved. Try a different industry.")
        st.stop()

    for u in urls[:5]:
        st.write(u)

    # -------------------------
    # Q3: Report generation
    # -------------------------
    if not api_key:
        st.error("Missing API key. Please set GOOGLE_API_KEY.")
        st.stop()

    user_prompt = build_user_prompt(industry, docs)

    with st.spinner("Generating industry report..."):
        report = generate_report(
            model_name=model_name,
            api_key=api_key,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt
        )

        report = enforce_under_500_words(report)
        report = ensure_sources_section(report, urls)

    st.subheader("Q3 — Industry Report (<500 words)")
    st.write(report)
    st.caption(f"Word count: {word_count(report)}")
