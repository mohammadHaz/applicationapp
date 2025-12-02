# python -m venv venv
# venv\Scripts\activate
# pip install streamlit pandas nltk scikit-learn pyarabic arabic-reshaper python-bidi

import streamlit as st
import pandas as pd
import nltk
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from pyarabic.araby import strip_diacritics, strip_tatweel, normalize_hamza
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
import string

# تحميل stopwords
nltk.download('stopwords')
arabic_stopwords = set(stopwords.words('arabic'))
arabic_stopwords.update(['في', 'على', 'إلى', 'من', 'هو','لا'])

# تنظيف النص
def clean_text(text):
    text = strip_diacritics(text)
    text = re.sub("[أإآا]", "ا", text)
    text = strip_tatweel(text)
    arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:".,'{}~¦+|!"…"–ـ'''
    english = string.punctuation
    all_p = set(arabic_punctuations + english)
    text = "".join([char if char not in all_p else " " for char in text])
    return text

# tokenization بدون camel-tools
def tokenize(text):
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)
    return text.split()

# stemming
stemmer = ISRIStemmer()
def stem_words(tokens):
    return [stemmer.stem(t) for t in tokens]

# حساب TF
def compute_tf(processed_sentences):
    tf_data = []
    all_terms = sorted(list(set([w for sent in processed_sentences for w in sent])))

    for i, tokens in enumerate(processed_sentences, 1):
        counts = Counter(tokens)
        total = len(tokens)
        row = {"Document": f"Doc{i}"}
        for term in all_terms:
            row[term] = round(counts[term] / total, 4) if total > 0 else 0
        tf_data.append(row)

    return pd.DataFrame(tf_data)

# Streamlit UI
st.title("📝 Arabic TF-IDF Analyzer)")

st.write("حل كامل لمعالجة النصوص العربية: تنظيف، تجزئة، إزالة الكلمات المحذوفة، Stemming، TF، TF-IDF")

input_option = st.radio("اختر طريقة الإدخال:", ["اكتب نصوص", "رفع CSV"])

sentences = []

if input_option == "اكتب نصوص":
    text_input = st.text_area("اكتب كل جملة في سطر:", height=200)
    if text_input:
        sentences = [s.strip() for s in text_input.split("\n") if s.strip()]

elif input_option == "رفع CSV":
    file = st.file_uploader("Upload CSV file", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.write("معاينة البيانات:")
        st.dataframe(df.head())
        column = st.selectbox("اختر العمود الذي يحتوي الجمل:", df.columns)
        sentences = df[column].astype(str).tolist()

if sentences:
    st.subheader("الجمل الأصلية")
    st.write(sentences)

    # 1- تنظيف
    cleaned = [clean_text(s) for s in sentences]
    st.subheader("1️⃣ الجمل بعد التنظيف")
    st.write(cleaned)

    # 2- Tokenization
    tokenized = [tokenize(s) for s in cleaned]
    st.subheader("2️⃣ Tokenization")
    st.write(tokenized)

    # 3- إزالة Stopwords
    filtered = [[w for w in tokens if w not in arabic_stopwords and len(w) > 1] for tokens in tokenized]
    st.subheader("3️⃣ بعد إزالة StopWords")
    st.write(filtered)

    # 4- Stemming
    stemmed = [stem_words(tokens) for tokens in filtered]
    st.subheader("4️⃣ بعد Stemming")
    st.write(stemmed)

    # 5- TF
    tf_df = compute_tf(stemmed)
    st.subheader("5️⃣ TF — Term Frequency")
    st.dataframe(tf_df)

    # 6- TF-IDF
    texts_joined = [" ".join(tokens) for tokens in stemmed]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts_joined)

    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=sorted(vectorizer.get_feature_names_out()),
        index=[f"Doc{i}" for i in range(1, len(sentences) + 1)]
    )

    st.subheader("6️⃣ TF-IDF")
    st.dataframe(tfidf_df)

    # 7- Top words
    st.subheader("🔝 أهم 5 كلمات في كل وثيقة")
    for doc in tfidf_df.index:
        st.write(f"### {doc}")
        st.write(tfidf_df.loc[doc].nlargest(5))