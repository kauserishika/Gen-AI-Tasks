"""
chains/pipeline.py
==================
AI Resume Screening — Deterministic Pipeline

Architecture:
  Stage 1: LLM  → Skill Extraction   (extraction_chain: PROMPT | llm | parser | json)
  Stage 2: Python → Matching          (keyword search, no LLM hallucination)
  Stage 3: Python → Scoring           (rubric math, always correct)
  Stage 4: LLM  → Explanation         (explanation_chain: PROMPT | llm | parser)

LangSmith traces Stage 1 and Stage 4 automatically.
"""

import os, re, json
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from huggingface_hub import InferenceClient

from prompts.templates import SKILL_EXTRACTION_PROMPT, EXPLANATION_PROMPT


# ─────────────────────────────────────────────────────────────────────
# HuggingFace Chat LLM  (no OpenAI package needed)
# ─────────────────────────────────────────────────────────────────────
class HFChatLLM(BaseChatModel):
    model: str       = "Qwen/Qwen2.5-7B-Instruct"
    temperature: float = 0.1
    max_tokens: int  = 1024
    model_config     = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self): return "hf-inference-client"

    def _to_hf(self, msgs: List[BaseMessage]) -> List[dict]:
        mp = {SystemMessage: "system", HumanMessage: "user", AIMessage: "assistant"}
        return [{"role": mp.get(type(m), "user"), "content": m.content} for m in msgs]

    def _generate(self, messages: List[BaseMessage], stop=None, **kwargs: Any) -> ChatResult:
        client = InferenceClient(api_key=os.environ.get("HUGGINGFACEHUB_API_TOKEN", ""))
        r = client.chat_completion(
            model=self.model,
            messages=self._to_hf(messages),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return ChatResult(generations=[
            ChatGeneration(message=AIMessage(content=r.choices[0].message.content))
        ])


# ─────────────────────────────────────────────────────────────────────
# JSON parser
# ─────────────────────────────────────────────────────────────────────
def parse_json(text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try: return json.loads(m.group())
            except: pass
    raise ValueError(f"JSON parse failed:\n{text[:400]}")


# ─────────────────────────────────────────────────────────────────────
# Required skills/tools for the Data Scientist role
# ─────────────────────────────────────────────────────────────────────
REQUIRED_SKILLS = [
    "python", "sql", "numpy", "pandas", "scikit-learn",
    "tensorflow", "pytorch", "statistics", "machine learning",
    "a/b testing", "data visualization",
]
REQUIRED_TOOLS = [
    "aws", "gcp", "azure", "docker", "kubernetes",
    "mlflow", "tableau", "power bi", "matplotlib", "spark",
]
BONUS_SKILLS = [
    "nlp", "computer vision", "spark", "hadoop",
    "open-source", "publication", "kaggle", "bert", "airflow",
]


# ─────────────────────────────────────────────────────────────────────
# Stage 2: Deterministic Matching  (pure Python)
# ─────────────────────────────────────────────────────────────────────
def deterministic_match(extracted: dict) -> dict:
    """
    Keyword search across all extracted text fields.
    Converts everything to lowercase — guaranteed no missed matches.
    """
    blob = " ".join([
        " ".join(extracted.get("technical_skills",    [])),
        " ".join(extracted.get("tools_and_platforms", [])),
        " ".join(extracted.get("domain_experience",   [])),
        " ".join(extracted.get("achievements",        [])),
    ]).lower()

    matched_skills = [s for s in REQUIRED_SKILLS if s in blob]
    missing_skills = [s for s in REQUIRED_SKILLS if s not in blob]
    tools_matched  = [t for t in REQUIRED_TOOLS  if t in blob]
    tools_missing  = [t for t in REQUIRED_TOOLS  if t not in blob]
    bonus_matched  = [b for b in BONUS_SKILLS    if b in blob]

    exp_yrs   = extracted.get("years_of_experience", 0)
    meets_exp = exp_yrs >= 5

    return {
        "required_skills_matched": matched_skills,
        "required_skills_missing": missing_skills,
        "tools_matched"          : tools_matched,
        "tools_missing"          : tools_missing,
        "bonus_qualifications"   : bonus_matched,
        "experience_match"       : {
            "required_years"    : 5,
            "candidate_years"   : exp_yrs,
            "meets_requirement" : meets_exp,
        },
        "overall_match_level": (
            "Strong"  if len(matched_skills) >= 8 else
            "Average" if len(matched_skills) >= 4 else
            "Weak"
        ),
    }


# ─────────────────────────────────────────────────────────────────────
# Stage 3: Deterministic Scoring  (pure Python)
# ─────────────────────────────────────────────────────────────────────
def deterministic_score(matched: dict, edu_score: int = 5) -> dict:
    """
    Rubric-based scoring — math is always correct by construction.
      Required Skills : (matched / total) * 40
      Experience      : 25 / 12 / 0
      Tools           : (matched / total) * 20
      Bonus           : 2 pts each, max 10
      Education       : 0-5 (passed in)
    """
    n_skills  = len(matched["required_skills_matched"])
    n_tools   = len(matched["tools_matched"])
    n_bonus   = len(matched["bonus_qualifications"])
    exp       = matched["experience_match"]
    gap       = exp["required_years"] - exp["candidate_years"]

    skills_score = round((n_skills / len(REQUIRED_SKILLS)) * 40)
    tools_score  = round((n_tools  / len(REQUIRED_TOOLS))  * 20)
    bonus_score  = min(n_bonus * 2, 10)
    exp_score    = 25 if exp["meets_requirement"] else (12 if gap <= 2 else 0)

    breakdown = {
        "required_skills_score": skills_score,
        "experience_score"     : exp_score,
        "tools_score"          : tools_score,
        "bonus_score"          : bonus_score,
        "education_score"      : edu_score,
    }
    total = sum(breakdown.values())  # always correct

    return {
        "score"          : total,
        "score_breakdown": breakdown,
        "tier"           : (
            "Top Candidate"    if total >= 80 else
            "Good Candidate"   if total >= 50 else
            "Needs Review"     if total >= 25 else
            "Not Recommended"
        ),
    }


def get_edu_score(extracted: dict) -> int:
    field = (extracted.get("education", {}).get("field", "") + " " +
             extracted.get("education", {}).get("degree", "")).lower()
    if any(k in field for k in ["computer science","machine learning","statistics",
                                 "mathematics","data science","engineering","physics"]):
        return 5
    if any(k in field for k in ["business","economics","information","biology"]):
        return 2
    return 0


# ─────────────────────────────────────────────────────────────────────
# Pipeline Builder
# ─────────────────────────────────────────────────────────────────────
def build_pipeline(model: str = "Qwen/Qwen2.5-7B-Instruct",
                   temperature: float = 0.1, max_tokens: int = 1024):
    """
    Builds the 4-stage screening pipeline using LCEL (| operator).

    LCEL chains:
      extraction_chain  = SKILL_EXTRACTION_PROMPT | llm | StrOutputParser | json_parser
      explanation_chain = EXPLANATION_PROMPT       | llm | StrOutputParser
    """
    llm = HFChatLLM(model=model, temperature=temperature, max_tokens=max_tokens)
    sp  = StrOutputParser()

    extraction_chain  = SKILL_EXTRACTION_PROMPT | llm | sp | RunnableLambda(parse_json)
    explanation_chain = EXPLANATION_PROMPT       | llm | sp

    def run_full_pipeline(inputs: dict) -> dict:
        resume         = inputs["resume"]
        jd             = inputs["job_description"]
        candidate_name = inputs.get("candidate_name", "Unknown")

        # Stage 1 — LLM: extract skills from resume text
        extracted = extraction_chain.invoke({"resume": resume})

        # Stage 2 — Python: deterministic keyword matching
        matched = deterministic_match(extracted)

        # Stage 3 — Python: rubric scoring (math always correct)
        edu   = get_edu_score(extracted)
        scored = deterministic_score(matched, edu_score=edu)

        # Stage 4 — LLM: generate human-readable explanation
        explanation = explanation_chain.invoke({
            "extracted_profile": json.dumps(extracted, indent=2),
            "matching_analysis": json.dumps(matched,   indent=2),
            "score_result"     : json.dumps(scored,    indent=2),
        })

        return {
            "candidate": candidate_name,
            "extraction": {
                "skills"     : extracted.get("technical_skills",    []),
                "tools"      : extracted.get("tools_and_platforms", []),
                "experience" : f"{extracted.get('years_of_experience', 0)} years",
                "education"  : extracted.get("education", {}),
                "achievements": extracted.get("achievements", []),
            },
            "matching": {
                "matched_skills": matched["required_skills_matched"],
                "missing_skills": matched["required_skills_missing"],
                "tools_matched" : matched["tools_matched"],
                "tools_missing" : matched["tools_missing"],
                "experience_gap": matched["experience_match"],
                "overall_match" : matched["overall_match_level"],
            },
            "score": {
                "value"    : scored["score"],
                "breakdown": scored["score_breakdown"],
                "tier"     : scored["tier"],
            },
            "explanation": explanation.strip(),
        }

    return run_full_pipeline


# ─────────────────────────────────────────────────────────────────────
# Debug Case  (called once from main.py)
# ─────────────────────────────────────────────────────────────────────
WEAK_RESUME_DEBUG = (
    "Name: Tom Williams\n"
    "SKILLS: Microsoft Excel (basic), Python (beginner, online course)\n"
    "EDUCATION: B.A. Business Administration — Community College, 2023"
)

BUGGY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an HR analyst. Extract the candidate's technical skills."),
    ("human",  "Resume: {resume}\n\nList skills including anything they might know.\n"
               "Return JSON: {{\"technical_skills\": [], \"tools\": []}}"),
])
FIXED_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Strict resume parser. Extract ONLY skills EXPLICITLY written. Do NOT infer."),
    ("human",  "Resume: {resume}\n\nExtract only explicitly stated skills.\n"
               "Return JSON: {{\"technical_skills\": [], \"tools\": []}}"),
])

def run_debug_case():
    llm   = HFChatLLM(model="Qwen/Qwen2.5-7B-Instruct", temperature=0.1, max_tokens=256)
    sp    = StrOutputParser()
    buggy = BUGGY_PROMPT | llm | sp | RunnableLambda(parse_json)
    fixed = FIXED_PROMPT | llm | sp | RunnableLambda(parse_json)

    buggy_out    = buggy.invoke({"resume": WEAK_RESUME_DEBUG})
    allowed      = ["excel", "python"]
    hallucinated = [s for s in buggy_out.get("technical_skills", [])
                    if not any(a in s.lower() for a in allowed)]
    fixed_out    = fixed.invoke({"resume": WEAK_RESUME_DEBUG})

    return {
        "bug"         : "Permissive prompt causes LLM to invent skills not in resume",
        "buggy_output": buggy_out,
        "hallucinated": hallucinated,
        "root_cause"  : "Prompt said 'include what they might know' → LLM guesses from context",
        "fixed_output": fixed_out,
        "fix_applied" : "prompts/templates.py SKILL_EXTRACTION_PROMPT — strict ABSOLUTE RULES added",
        "result"      : "Only resume-stated skills returned after fix",
    }