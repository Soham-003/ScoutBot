import os
import streamlit as st
import pickle
import time
import requests
from bs4 import BeautifulSoup
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()  # Load .env (especially OPENAI_API_KEY)
# Inject CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("ScoutBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = [st.sidebar.text_input(f"URL {i + 1}") for i in range(3)]
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"
main_placeholder = st.empty()
llm = OpenAI(model_name="gpt-4o-mini",temperature=0.9, max_tokens=500)

def fetch_article_text(url):
    """Fetches article text using requests + BeautifulSoup."""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Generic selector: all paragraphs
        text = "\n".join(p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True))
        return Document(page_content=text, metadata={"source": url}) if text else None
    except Exception as e:
        st.warning(f"Failed to load {url}: {e}")
        return None

if process_url_clicked:
    # Clean URLs and validate
    urls = [u.strip() for u in urls if u and u.strip()]
    if not urls:
        st.error("⚠ Please enter at least one valid news article URL.")
        st.stop()

    main_placeholder.text("Data Loading... Started... ✅")
    docs_raw = [doc for url in urls if (doc := fetch_article_text(url))]

    if not docs_raw:
        st.error("⚠ No data could be loaded from the provided URLs. Please check the links.")
        st.stop()

    main_placeholder.text("Text Splitter... Started... ✅")
    splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    docs = splitter.split_documents(docs_raw)
    if not docs:
        st.error("⚠ No text was extracted from the loaded pages.")
        st.stop()

    main_placeholder.text("Building Embedding Vector... Started... ✅")
    embeddings = OpenAIEmbeddings()
    try:
        vectorstore = FAISS.from_documents(docs, embeddings)
    except Exception as e:
        st.error(f"❌ Error creating FAISS index: {e}")
        st.stop()

    time.sleep(1)
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)
   

# Query Section
query = main_placeholder.text_input("Question: ")
# When querying:
if query and os.path.exists(file_path):
    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",  # avoids multi-prompt issue
        return_source_documents=True
    )

    result = chain({"query": query})

    st.header("Answer")
    st.write(result["result"])

    # sources = result.get("source_documents", [])
    # if sources:
    #     st.subheader("Sources:")
    #     for doc in sources:
    #         st.write(doc.metadata.get("source", "Unknown"))