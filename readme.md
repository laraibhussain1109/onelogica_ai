## Streamlit Resume Ranker

A local Streamlit application to automatically rank candidate resumes based on a job description, using a combination of semantic similarity, keyword extraction, and experience matching.


### Features

- **Automatic Keyword Extraction**: Uses TF-IDF to extract top N keywords from the job description.
- **Semantic Similarity**: Leverages `SentenceTransformer('all-MiniLM-L6-v2')` to encode both job description and resumes into dense vectors and computes cosine similarity.
- **Keyword Match Score**: Counts occurrences of extracted keywords in each resume.
- **Experience Match Score**: Parses phrases like `X years` and scores based on desired minimum experience.
- **Weighted Scoring**: Allows customizable weights for semantic, keyword, and experience components.
- **Local Folder Processing**: Specify a local folder path; the app scans for `.pdf` and `.docx` resumes.

---

### Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/Mac
   venv\Scripts\activate       # Windows
   ```
2. Install dependencies:
   ```bash
   pip install streamlit sentence-transformers scikit-learn PyPDF2 python-docx pandas
   ```
3. Save the main script as streamlit run onelogica.py.

---

### Usage

1. Run the app:
   ```bash
   streamlit run mega_testing.py
   ```
2. In the sidebar:
   - Upload your `job_description.txt` file.
   - Set desired minimum years of experience and scoring weights.
   - Choose how many keywords to extract.
   - Enter the full path to your local folder containing resumes.
3. View the ranked results and score breakdown in the main panel.
