"""
main.py — AI Resume Screening System
HuggingFace + LangSmith | Structured JSON Output | No OpenAI

"""

import os, json, warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT",    "resume-screening-system")
os.environ.setdefault("LANGCHAIN_ENDPOINT",   "https://api.smith.langchain.com")

if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
    print("❌HUGGINGFACEHUB_API_TOKEN missing → https://huggingface.co/settings/tokens")
    raise SystemExit(1)

from chains.pipeline import build_pipeline, run_debug_case
from data.job_description import JOB_DESCRIPTION, RESUMES


def main():
    pipeline    = build_pipeline(model="Qwen/Qwen2.5-7B-Instruct", temperature=0.1)
    all_results = []

    # ── Run debug case (returns JSON, no print noise) ─────────────────
    debug_result = run_debug_case()

    # ── Screen all 3 candidates ───────────────────────────────────────
    for label, resume_text in RESUMES.items():
        result = pipeline({
            "resume"         : resume_text,
            "job_description": JOB_DESCRIPTION,
            "candidate_name" : f"{label} Candidate",
        })
        all_results.append(result)

    # ── Final structured JSON output (single clean block) ─────────────
    final_output = {
        "system"    : "AI Resume Screening System",
        "model"     : "Qwen/Qwen2.5-7B-Instruct (HuggingFace Serverless)",
        "tracing"   : f"LangSmith — project: resume-screening-system",
        "job_role"  : "Senior Data Scientist @ TechCorp Analytics",
        "debug_case": debug_result,
        "candidates": all_results,
        "ranking"   : [
            {
                "rank"     : i + 1,
                "candidate": r["candidate"],
                "score"    : r["score"]["value"],
                "tier"     : r["score"]["tier"],
            }
            for i, r in enumerate(
                sorted(all_results, key=lambda x: x["score"]["value"], reverse=True)
            )
        ],
    }

    print(json.dumps(final_output, indent=2))

    with open("results.json", "w") as f:
        json.dump(final_output, f, indent=2)


if __name__ == "__main__":
    main()