"""
prompts/templates.py
LLM is used ONLY for: Skill Extraction + Explanation
Matching and Scoring are handled by deterministic Python (no LLM).
"""
from langchain_core.prompts import ChatPromptTemplate

# ── Stage 1: Skill Extraction (LLM) ──────────────────────────────────
SKILL_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict resume parser. Extract ONLY information explicitly written in the resume.\n"
     "RULES:\n"
     "- Do NOT infer skills from job titles\n"
     "- Do NOT add skills the candidate 'might know'\n"
     "- If a skill is not written word-for-word, leave it out\n"
     "- Return ONLY valid JSON, no markdown, no explanation"),
    ("human",
     "Resume:\n{resume}\n\n"
     "Return this exact JSON:\n"
     "{{\n"
     '  "technical_skills": ["list each skill/library individually"],\n'
     '  "tools_and_platforms": ["list each tool individually"],\n'
     '  "years_of_experience": 0,\n'
     '  "domain_experience": ["domain1"],\n'
     '  "education": {{"degree": "", "field": "", "institution": ""}},\n'
     '  "achievements": ["achievement1"]\n'
     "}}")
])

# ── Stage 4: Explanation (LLM) ────────────────────────────────────────
EXPLANATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an HR consultant writing a factual candidate evaluation.\n"
     "RULES:\n"
     "- Only mention skills that appear in matched_skills or extracted profile\n"
     "- Do NOT contradict the score or matching data\n"
     "- 3-5 sentences, plain text only\n"
     "- Start with: 'This candidate scored X/100...'"),
    ("human",
     "Extracted Profile:\n{extracted_profile}\n\n"
     "Matching Result:\n{matching_analysis}\n\n"
     "Score:\n{score_result}")
])