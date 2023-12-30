import streamlit as st
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity

# Fungsi untuk melakukan crawling data dari URL jurnal
def crawl_journal_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # Misalnya, kita mengambil teks dari tag <p> pada halaman jurnal
    text_data = " ".join([p.get_text() for p in soup.find_all('p')])
    return text_data

# Fungsi untuk melakukan preprocessing data
def preprocess_data(text_data):
    # Implementasikan preprocessing sesuai kebutuhan Anda
    # Misalnya, konversi ke huruf kecil, penghapusan karakter khusus, dll.
    processed_data = text_data.lower()
    return processed_data

# Fungsi untuk melatih model SVM
def train_svm_model(features, labels):
    svm_model = SVC(kernel='linear')
    svm_model.fit(features, labels)
    return svm_model

# Fungsi untuk melakukan pencarian dengan VSM
def search_with_vsm(data, query, documents):
    # Menggunakan TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([query] + documents)
    
    # Menghitung skor cosine similarity
    similarity_scores = cosine_similarity(vectors[0], vectors[1:]).flatten()
    
    # Menyusun hasil pencarian
    search_results = list(zip(documents, similarity_scores))
    search_results.sort(key=lambda x: x[1], reverse=True)
    
    return search_results

# Antarmuka Pengguna dengan Streamlit
st.title("Crawling Jurnal dan Pencarian Kata dengan VSM")

# Input URL dari pengguna
url = st.text_input("Masukkan URL Jurnal:")
if url:
    # Crawling data dari URL jurnal
    data = crawl_journal_data(url)

    # Preprocessing data
    processed_data = preprocess_data(data)

    # Input pencarian dari pengguna
    query = st.text_input("Masukkan kata kunci pencarian:")

if st.button("Cari"):
    if query:
        # Pencarian dengan VSM
        search_results = search_with_vsm(processed_data, query, [processed_data])

        # Menampilkan hasil pencarian
        st.subheader("Hasil Pencarian:")
        for result, score in search_results:
            st.write(f"Skor: {score:.4f}")
            st.write(result)