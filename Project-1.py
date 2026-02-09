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
    Validates if the input is a valid industry by checking Wikipedia retrieval.
    Verifies that results are industry-related, not random pages or products.
    If invalid, suggests similar industry terms.
    
    Returns: (is_valid: bool, suggestion: str or None)
    """
    def is_disambiguation(title: str, content: str) -> bool:
        title_l = title.lower()
        content_l = content.lower()
        if "(disambiguation)" in title_l:
            return True
        # Wikipedia disambiguation pages usually contain this phrase
        if "may refer to" in content_l[:400]:
            return True
        return False

    # Common products/objects that are NOT industries
    non_industries = [
        "glasses", "coffee", "water", "pizza", "apple", "car", "house",
        "phone", "book", "pen", "table", "chair", "shoes", "shirt",
        "hat", "dog", "cat", "bicycle", "tree", "flower", "rock"
    ]
    
    input_lower = industry.lower().strip()
    
    # Quick check: reject common non-industry terms
    if input_lower in non_industries:
        suggestion = find_similar_industry(industry)
        return False, suggestion
    
    # Keywords that suggest an industry/business context (positive signals)
    industry_keywords = [
        "industry", "market", "business", "sector", "trade", "commerce",
        "manufacturing", "production", "service", "company", "enterprise",
        "economic", "technology", "economy",
        "supply", "demand", "consumer", "corporate", "commercial",
        "retail", "finance", "energy", "construction", "transportation",
        "logistics", "insurance", "banking", "healthcare", "pharmaceutical",
        "telecommunications", "media", "hospitality", "agriculture"
    ]

    # Keywords that usually indicate a specific product/topic (negative signals)
    non_industry_keywords = [
        "software", "app", "device", "product", "brand", "model", "game",
        "song", "film", "novel", "restaurant", "city", "country", "person",
        "species", "plant", "animal", "weapon", "car", "phone", "laptop"
    ]

    # Phrases in lead sentences that usually indicate a single entity (not an industry)
    entity_lead_phrases = [
        "is a company", "is an american company", "is a multinational",
        "is a corporation", "is a brand", "is a product", "is a device",
        "is a software", "is an app", "is a city", "is a town",
        "is a country", "is a person", "is a musician", "is a singer",
        "is a film", "is a novel", "is a video game", "is a car",
        "is a model", "is a phone", "is an animal", "is a species"
    ]
    
    retriever = WikipediaRetriever(top_k_results=top_k, lang="en")

    # Build alternative query terms to improve recall for inputs like "camera industry"
    def build_query_terms(raw: str) -> list:
        term = raw.lower().strip()
        terms = [raw]
        for kw in ["industry", "sector", "market"]:
            if kw in term:
                base = term.replace(kw, "").strip()
                if base:
                    terms.append(base)
                    terms.append(f"{base} market")
                    terms.append(f"{base} manufacturing")
                    terms.append(f"{base} business")
                break
        return list(dict.fromkeys(terms))

    query_terms = build_query_terms(industry)

    # Retrieve and merge docs across query terms
    docs = []
    seen = set()
    for q in query_terms:
        for d in retriever.invoke(q):
            key = (d.metadata.get("source", ""), d.metadata.get("title", ""))
            if key not in seen:
                seen.add(key)
                docs.append(d)
    
    if not docs:
        # No results found - invalid input
        suggestion = find_similar_industry(industry)
        return False, suggestion
    
    # Check if results are industry-related by looking at titles and content
    industry_score = 0
    strong_title_signal = 0
    total_docs = len(docs)

    def lead_sentence(text: str) -> str:
        snippet = re.sub(r"\s+", " ", text.strip())[:300]
        if "." in snippet:
            return snippet.split(".", 1)[0].lower()
        return snippet.lower()
    
    for doc in docs:
        title = doc.metadata.get("title", "").lower()
        content = doc.page_content.lower()

        # Skip disambiguation pages (they usually don't define industries)
        if is_disambiguation(title, content):
            continue
        
        # Count industry-related keywords in title (title has more weight)
        title_keyword_matches = sum(1 for keyword in industry_keywords if keyword in title)
        content_keyword_matches = sum(1 for keyword in industry_keywords if keyword in content[:600])
        
        # Count non-industry signals
        negative_title_matches = sum(1 for keyword in non_industry_keywords if keyword in title)
        negative_content_matches = sum(1 for keyword in non_industry_keywords if keyword in content[:600])

        # Penalize pages that look like a single entity (company/product/place/person)
        lead = lead_sentence(doc.page_content)
        entity_lead_matches = sum(1 for p in entity_lead_phrases if p in lead)
        negative_content_matches += entity_lead_matches * 2
        
        # Title keywords are weighted more heavily
        total_matches = (title_keyword_matches * 2) + content_keyword_matches
        total_negatives = (negative_title_matches * 2) + negative_content_matches
        
        # Only count as industry-related if positives outweigh negatives
        if total_matches > total_negatives and total_matches > 0:
            industry_score += 1
            if any(k in title for k in ["industry", "sector", "market"]):
                strong_title_signal += 1
    
    # Prefer LLM-based validation when available
    if api_key and model_name:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.0,
                max_output_tokens=200
            )

            # Build a compact evidence block from the top docs
            evidence = []
            for d in docs[:3]:
                title = d.metadata.get("title", "Unknown title")
                extract = d.page_content.strip()
                extract = extract[:600]
                evidence.append(f"- {title}: {extract}")

            evidence_text = "\n".join(evidence)

            llm_prompt = f"""
You are classifying whether a term is an INDUSTRY (not a product, brand, person, place, or specific company).
Given the term and Wikipedia evidence, respond ONLY in JSON with keys:
{{"is_industry": "yes" or "no", "reason": "short reason"}}

Term: {industry}

Wikipedia evidence:
{evidence_text}
""".strip()

            result = llm.invoke([("user", llm_prompt)])
            content = result.content
            if isinstance(content, dict):
                content = content.get("text", str(content))
            elif isinstance(content, list):
                content = "".join([str(c) for c in content])

            text = content.strip()
            # Simple JSON parse without external deps
            is_industry_match = re.search(r'"is_industry"\\s*:\\s*"(yes|no)"', text, re.IGNORECASE)
            if is_industry_match:
                llm_is_industry = is_industry_match.group(1).lower() == "yes"
                if llm_is_industry:
                    return True, None
                else:
                    suggestion = find_similar_industry(industry)
                    return False, suggestion
        except Exception:
            # If LLM validation fails, fall back to heuristic
            pass

    # Heuristic fallback (when no API key/model or LLM fails)
    # - At least 50% of results industry-related
    # - At least 1 strong title signal OR some base-term match
    validity_threshold = max(1, int(total_docs * 0.5))
    heuristic_valid = industry_score >= validity_threshold and (strong_title_signal > 0 or base_term_matches > 0)

    # If user explicitly typed "industry/sector/market", keep same thresholds
    if any(k in input_lower for k in ["industry", "sector", "market"]):
        heuristic_valid = heuristic_valid or (industry_score > 0 and (strong_title_signal > 0 or base_term_matches > 0))

    if heuristic_valid:
        return True, None
    else:
        # Results don't seem industry-related
        suggestion = find_similar_industry(industry)
        return False, suggestion


def find_similar_industry(invalid_input: str) -> str:
    """
    Suggests relevant industry terms based on the user's input.
    Uses:
    - A small synonym/related-industry map
    - Fallback string similarity on a common industry list
    Returns top suggestions for the user.
    """
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

    st.subheader("Q3 — Industry Report (<500 words)")
    st.write(report)
    st.caption(f"Word count: {word_count(report)}")
