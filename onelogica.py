import streamlit as st
import tempfile
import os
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import PyPDF2
import docx2txt
import pandas as pd

# Streamlit app configuration
st.set_page_config(page_title="Laraib's Resume Ranker", layout="wide")
st.title("Laraib's Resume Ranker - Best Fit for Your Job Description")

# Yeh hai streamlit app for Resume Ranker
# yahan se bas import kar k shuru kiya hai configuration

# Dhyaan rakhna bhai isko start karne k liye `streamlit run mega_testing.py` command use karna hai

# Hume pta hai tum isko padhoge nhi fir bhi likh dete hain
# Yeh sentence_transformers model ko use karta hai jo ki pehle se hi prachalit hai Resume screening mein
# dhyaan rahe ki yeh minimal model hai, agar zyada accuracy chahiye toh tum `all-mpnet-base-v2` ya `all-distilroberta-v1` use kar sakte ho
# uske liye bas `SentenceTransformer('all-mpnet-base-v2')` ya `SentenceTransformer('all-distilroberta-v1')` karna padega
# yeh cache_resource decorator se model ko cache karte hain taaki baar baar load na ho
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# yeh sidebar diya hai bas aise hi taki user ko job description aur ranking parameters set kar sake
# iska title bhi set kar skte hain st.sidebar.title("") lekin mann nhi hai isiliye nhi likha
st.sidebar.header("Job Description & Ranking Parameters")
job_file = st.sidebar.file_uploader("Upload job_description.txt (TXT)", type=['txt'])

# Apne according weights set kar do bas aur kya
desired_exp = st.sidebar.number_input(
    "Desired minimum years of experience", min_value=0, max_value=50, value=1
)
st.sidebar.markdown("---")
st.sidebar.subheader("Weights for Scoring")
w_sem = st.sidebar.slider("Semantic Similarity Weight", 0.0, 1.0, 0.5)
w_key = st.sidebar.slider("Keyword Match Weight", 0.0, 1.0, 0.3)
w_exp = st.sidebar.slider("Experience Match Weight", 0.0, 1.0, 0.2)
#normalization toh pta hi hoga.... Itni NLP toh sabko aati hai
# total weight ko normalize karte hain taki sum 1 ho jaye
total_w = w_sem + w_key + w_exp
w_sem /= total_w; w_key /= total_w; w_exp /= total_w

# Yeh keywords auto-extraction k liye code hai yeh jo uploaded job description hoga usse keywords nikaal lega
st.sidebar.markdown("---")
st.sidebar.subheader("Auto-extract Keywords")
auto_n = st.sidebar.slider("Number of keywords to extract", 1, 10, 5)

# Folder ka path de do jaldi se
st.sidebar.markdown("---")
st.sidebar.subheader("Resume Folder Path")
folder_selected = st.sidebar.text_input("Enter full path to folder containing resumes")
if folder_selected and not os.path.isdir(folder_selected):
    st.sidebar.error("Folder path is invalid. Please enter a valid directory.")

if job_file and folder_selected and os.path.isdir(folder_selected):
    job_description = job_file.read().decode('utf-8')
    st.sidebar.success("Job description loaded")
    st.sidebar.markdown("**Preview:**\n```" + job_description[:300] + "...```")

    # Keyword extraction: TF-IDF
    def extract_keywords(text, top_n):
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf = vectorizer.fit_transform([text])
        scores = dict(zip(vectorizer.get_feature_names_out(), tfidf.toarray()[0]))
        sorted_terms = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [term for term, score in sorted_terms[:top_n]]

    extracted_keywords = extract_keywords(job_description, auto_n)
    st.sidebar.markdown(f"**Extracted keywords:** {', '.join(extracted_keywords)}")

    keywords = [kw.lower() for kw in extracted_keywords]
    job_emb = model.encode([job_description])[0]

    # Poora ka poora folder ka naam de do fully qualified path ke saath
    st.header("Processing Resumes from Folder")
    results = []
    for file_name in os.listdir(folder_selected):
        if file_name.lower().endswith(('.pdf', '.docx')):
            file_path = os.path.join(folder_selected, file_name)
            # text Nikalne ki ninja technique
            if file_name.lower().endswith('pdf'):
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ''.join([page.extract_text() or '' for page in reader.pages])
            else:
                text = docx2txt.process(file_path)
            text_low = text.lower()

            ## final score ka calculation
            sem_score = float(cosine_similarity([job_emb], [model.encode([text])[0]])[0][0])
            key_score = sum(text_low.count(kw) for kw in keywords) / len(keywords) if keywords else 0.0
            exp_matches = re.findall(r"(\d+)\+?\s+years?", text_low)
            exp_years = max(map(int, exp_matches)) if exp_matches else 0
            exp_score = min(exp_years / desired_exp, 1.0) if desired_exp > 0 else 0.0
            final_score = w_sem * sem_score + w_key * key_score + w_exp * exp_score

            results.append({
                'Resume': file_name,
                'Semantic': sem_score,
                'Keywords': key_score,
                'Experience': exp_score,
                'Score': final_score
            })

    # ab itni mehnat kari hai toh results bhi dekh hi lo wahi apna purana pandas ka dataframe hai
    df = pd.DataFrame(results).sort_values('Score', ascending=False).reset_index(drop=True)
    df.index += 1; df.index.name = 'Rank'
    st.subheader("Ranking Results")
    st.table(df.style.format({
        'Semantic': '{:.3f}',
        'Keywords': '{:.2f}',
        'Experience': '{:.2f}',
        'Score': '{:.3f}'
    }))
    st.subheader("Score Breakdown")
    st.bar_chart(df[['Semantic', 'Keywords', 'Experience', 'Score']])
else:
    st.info("Please upload a job_description.txt file and specify a valid resume folder to begin.")



# Yahan apna naam likh dete hain heart emoji copy kar k
st.markdown("---")
st.markdown("Built with ❤️ by Laraib Hussain.")
