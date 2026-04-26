🚀 AI Resume Screening System
HuggingFace + LangChain + LangSmith Tracing

📌Overview

This project is an AI-powered Resume Screening System designed to automate candidate evaluation for a Senior Data Scientist role.

It uses LLMs via HuggingFace Inference API and a 4-stage LangChain pipeline to:
1. Extract candidate skills
2. Match them against job requirements
3. Score candidates using a strict rubric
4. Generate professional hiring recommendations

Pipeline Architecture:
Resume → Extract → Match → Score → Explain


🎯Key Features: 
* Strict Skill Extraction (No Hallucination)
* Rule-based Matching System
* Deterministic Scoring (0–100)
* LLM-powered Explanation Generation
* LangSmith Tracing 
* Debug Case Demonstration

🔗 Pipeline Flow (LCEL): 
extraction_chain  = SKILL_EXTRACTION_PROMPT | llm | parser
matching_chain    = MATCHING_PROMPT         | llm | parser
scoring_chain     = SCORING_PROMPT          | llm | parser | validate_score
explanation_chain = EXPLANATION_PROMPT      | llm

📂 Project Structure:

ai-resume-screening/
│
├── chains/
│   └── pipeline.py         
│
├── prompts/
│   └── templates.py         
│
├── data/
│   └── job_description.py   
│
├── main.py                 
├── results.json             
├── .env                     

🔑 Setup Instructions

1. Install Dependencies
pip install -r requirements.txt

2. Add Environment Variables

3. Create .env file:
HUGGINGFACEHUB_API_TOKEN=your_hf_token
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=resume-screening-system

4. Run Project
python main.py

⚠️ Challenges Faced:

* LLM hallucination in extraction
* Incorrect scoring outputs
* Skill matching inconsistencies

✅ Solutions Implemented:

* Strict prompt engineering
* JSON-only outputs enforced
* Score validation function
* Deterministic rubric

📈Future Improvements:

* Add UI (Streamlit / React)
* Resume upload (PDF parsing)
* Multi-role support
* Real-time API deployment