import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter
import re
import json
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from openai import OpenAI

# Initialize models only once
@st.cache_resource
def load_models():
    model_name = 'sentence-transformers/paraphrase-xlm-r-multilingual-v1'
    model = SentenceTransformer(model_name)
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return model, cross_encoder

model, cross_encoder = load_models()

# Sidebar for inputs
st.sidebar.title("YouTube Video Summarizer & Q&A")
video_url = st.sidebar.text_input("Enter YouTube Video URL", key="video_url")
api_key = st.sidebar.text_input("Enter OpenAI API Key (optional)", type="password")

# Initialize session state variables
if "api_key" not in st.session_state:
    st.session_state["api_key"] = api_key

# Function to extract video ID from the URL
def get_video_id(url):
    match = re.search(r'v=([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid YouTube URL")

# Function to process video and get transcript
@st.cache_data
def process_video(url):
    try:
        video_id = get_video_id(url)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        formatter = JSONFormatter()
        transcript_json = json.loads(formatter.format_transcript(transcript))
        all_text = " ".join([pt["text"] for pt in transcript_json])

        with open('output.txt', 'w') as file:
            file.write(all_text)

        return all_text
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None

# Function to chunk and embed the video text
@st.cache_data
def chunk_embed(text, chunk_size=400):
    loader = TextLoader('output.txt')
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    document_embeddings = model.encode([chunk.page_content for chunk in chunks])
    return document_embeddings, [chunk.page_content for chunk in chunks]

# Function to get GPT response using the updated OpenAI API
def get_response(prompt):
    client = OpenAI(api_key=st.session_state["api_key"])
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-4o-mini",
    )
    return chat_completion.choices[0].message.content.strip()

# Function to answer questions based on selected ranking method
def answer_question(query, top_k, option):
    query_embedding = model.encode(query)
    similarities = cosine_similarity([query_embedding], st.session_state["video_chunk_embed"])
    sorted_indices = np.argsort(similarities[0])[::-1]
    ranked_documents = [(st.session_state["video_chunks"][i], similarities[0][i]) for i in sorted_indices[:top_k]]

    if option == "No_ReRanking":
        return ranked_documents
    elif option == "Cross_Encoder":
        pairs = [[query, doc] for doc, _ in ranked_documents]
        scores = cross_encoder.predict(pairs)
        scored_docs = sorted(zip([doc for doc, _ in ranked_documents], scores), key=lambda x: x[1], reverse=True)
        return scored_docs 
    else:  # BM25
        top_k_documents = [doc[0] for doc in ranked_documents[:top_k]]
        tokenized_top_k_documents = [doc.split() for doc in top_k_documents]
        tokenized_query = query.split()
        bm25 = BM25Okapi(tokenized_top_k_documents)
        bm25_scores = bm25.get_scores(tokenized_query)
        sorted_indices2 = np.argsort(bm25_scores)[::-1]
        reranked_documents = [(top_k_documents[i], bm25_scores[i]) for i in sorted_indices2]
        return reranked_documents

# Function to generate response using GPT
def gen_rsp(question):
    context = "\n".join([doc for doc, _ in st.session_state[st.session_state["opt"]]])
    prompt = f"""Answer the User question, taking help from the context provided.\n 
    context : 
    {context}
    \n\n
    Question : 
    {question}
    """
    return get_response(prompt)

# Function to generate summary using GPT
def gen_sum_rsp():
    prompt = f"""Generate the summary of the given transcript of a video\n{st.session_state["video_text"]}"""
    return get_response(prompt)

# Main content
st.title("YouTube Video Summarizer & Q&A")

# Store previous video URL
if "prev_video_url" not in st.session_state:
    st.session_state["prev_video_url"] = ""

# Check if the URL has changed
url_changed = st.session_state["prev_video_url"] != video_url

if url_changed:
    st.session_state["video_text"] = None
    st.session_state["video_chunk_embed"] = None
    st.session_state["video_chunks"] = None
    st.session_state["prev_video_url"] = video_url
    st.session_state["opt"]=None
    st.session_state["No_ReRanking"]=None
    st.session_state["Cross_Encoder"]=None
    st.session_state["BM25"]=None

if st.sidebar.button("Process Video") and video_url:
    with st.spinner("Processing video..."):
        st.session_state["video_text"] = process_video(video_url)
        if st.session_state["video_text"]:
            st.session_state["video_chunk_embed"], st.session_state["video_chunks"] = chunk_embed(st.session_state["video_text"])
            st.success("Video processed successfully!")
        else:
            st.error("Failed to process video.")

if st.session_state.get("video_text"):
    st.subheader("Video Summary & Query")

    if api_key:
        with st.expander("Generate Summary"):
            if st.button("Generate Summary"):
                with st.spinner("Generating summary..."):
                    summary = gen_sum_rsp()
                    st.write(summary)
    else:
        st.info("Enter your OpenAI API key in the sidebar to enable summary generation.")

    with st.expander("Ask a Question"):
        question = st.text_input("Enter your question")
        top_k = st.slider('Select Top K Documents', min_value=1, max_value=10, value=5)
        option = st.selectbox('ReRanking Method', ('No_ReRanking', 'Cross_Encoder', 'BM25'))

        if question and top_k > 0 and option:
            st.session_state["opt"] = option
            ranked_docs = answer_question(question, top_k, option)
            st.session_state[option] = ranked_docs
            for i, (doc, score) in enumerate(ranked_docs[:top_k]):
                st.markdown(f"**Rank {i + 1}:**\n- **Doc:** {doc}\n- **Score:** {score:.4f}\n---")

            if api_key:
                if st.button("Generate LLM Response"):
                    with st.spinner("Generating response..."):
                        st.write(gen_rsp(question))
            else:
                st.info("Enter your OpenAI API key in the sidebar to enable LLM-based Q&A.")
        else:
            st.error("Please enter a question and select a ranking method.")
