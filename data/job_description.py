"""
Job Description and Resume Data
Contains the job description and three sample resumes (Strong, Average, Weak)
for a Data Scientist role.
"""

JOB_DESCRIPTION = """
Position: Senior Data Scientist
Company: TechCorp Analytics

Requirements:
- 5+ years of experience in data science or machine learning
- Proficiency in Python (NumPy, Pandas, Scikit-learn, TensorFlow/PyTorch)
- Strong SQL skills for data querying and manipulation
- Experience with cloud platforms (AWS, GCP, or Azure)
- Knowledge of statistical modeling and A/B testing
- Experience with data visualization tools (Tableau, Power BI, or Matplotlib/Seaborn)
- Familiarity with MLOps practices (MLflow, Docker, Kubernetes)
- Experience building and deploying production ML models
- Strong communication skills to present findings to stakeholders

Nice-to-Have:
- NLP or Computer Vision experience
- Experience with Spark or distributed computing
- Publications or contributions to open-source projects
"""

RESUMES = {
    "Strong": """
Name: Sarah Chen
Email: sarah.chen@email.com

SUMMARY:
Senior Data Scientist with 7 years of experience building production-grade ML systems.
Passionate about turning complex data into actionable business insights.

EXPERIENCE:
Senior Data Scientist — DataDriven Inc. (2020–Present)
- Led a team of 4 data scientists to build a real-time fraud detection system using XGBoost and PyTorch, reducing fraud losses by 35%
- Deployed ML models to AWS SageMaker serving 2M+ predictions/day
- Ran 50+ A/B tests to optimize product recommendations, increasing CTR by 22%
- Built NLP pipeline using BERT for customer support ticket classification (94% accuracy)

Data Scientist — Analytics Corp (2017–2020)
- Developed customer churn prediction models using Scikit-learn and Pandas
- Created interactive Tableau dashboards for C-suite reporting
- Optimized SQL queries processing 500GB+ daily data on Redshift

SKILLS:
Python (Expert): NumPy, Pandas, Scikit-learn, TensorFlow, PyTorch, Matplotlib, Seaborn
Cloud: AWS (SageMaker, S3, EC2, Lambda), GCP (BigQuery)
MLOps: MLflow, Docker, Kubernetes, CI/CD pipelines
Databases: SQL (PostgreSQL, Redshift, MySQL), NoSQL (MongoDB)
Other: Spark, Hadoop, Git, Airflow, NLP, Computer Vision

EDUCATION:
M.S. Computer Science (Machine Learning focus) — Stanford University, 2017
B.S. Mathematics & Statistics — UC Berkeley, 2015

ACHIEVEMENTS:
- Published 2 papers in NeurIPS and ICML on federated learning
- Maintainer of open-source ML toolkit with 3,000+ GitHub stars
- Winner of Kaggle Competition (top 1%)
""",

    "Average": """
Name: Marcus Johnson
Email: marcus.j@email.com

SUMMARY:
Data Scientist with 3 years of experience in analytics and machine learning.
Looking for opportunities to grow in a fast-paced environment.

EXPERIENCE:
Data Scientist — RetailCo (2022–Present)
- Built sales forecasting models using Python and Scikit-learn
- Created weekly reports using Power BI for business stakeholders
- Wrote SQL queries to extract data from company databases
- Assisted senior data scientists with A/B test analysis

Junior Data Analyst — StartupXYZ (2021–2022)
- Analyzed marketing data using Python and Excel
- Created visualizations using Matplotlib
- Maintained data pipelines using basic SQL

SKILLS:
Python: Pandas, NumPy, Scikit-learn, Matplotlib
SQL: MySQL, basic PostgreSQL
Visualization: Power BI, Matplotlib
Some exposure to: AWS S3, Docker (beginner)

EDUCATION:
B.S. Statistics — State University, 2021

ACHIEVEMENTS:
- Improved sales forecast accuracy by 12%
- Completed AWS Cloud Practitioner certification
""",

    "Weak": """
Name: Tom Williams
Email: tom.w@email.com

SUMMARY:
Recent graduate interested in data science. Eager to learn and contribute.

EXPERIENCE:
Data Entry Intern — LocalBiz (Summer 2023)
- Entered data into spreadsheets
- Helped organize company files
- Assisted with basic Excel reporting

Part-time Cashier — Grocery Store (2021–2023)
- Handled cash transactions
- Provided customer service

SKILLS:
Microsoft Excel (basic)
Python (beginner, completed online course)
Basic statistics from university coursework

EDUCATION:
B.A. Business Administration — Community College, 2023

CERTIFICATIONS:
- Coursera Python for Everybody (in progress)
"""
}