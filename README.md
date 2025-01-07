# **YouTube Video Summarizer & Q&A Chatbot**

This project allows users to summarize YouTube videos and ask questions related to the content of the video. The application uses **Streamlit**, **YouTube Transcript API**, and **Google Gemini** models to provide video summaries and answers to questions based on the video’s transcript.



## **Overview**

The **YouTube Video Summarizer & Q&A** application provides the following features:

- **Summarize YouTube Video:** Automatically summarize the transcript of a YouTube video.
- **Ask Questions:** You can ask specific questions based on the transcript, and the system will rank the relevant sections and generate answers using GPT-based models.
- **Re-Ranking Options:** Use different methods like **No Re-Ranking**, **Cross-Encoder**, and **BM25** for document ranking before generating answers.

---

## **Features**

- **Video Transcript Extraction:** Extracts transcripts of YouTube videos using the **YouTube Transcript API**.
- **Text Chunking & Embedding:** Splits the video transcript into chunks and generates embeddings using **Sentence Transformers**.
- **Question Answering:** Allows users to ask questions about the video’s content and ranks relevant sections using various ranking methods.
- **Summarization:**  Uses **Google Gemini** models to generate a summary of the video.

---

## **Installation**

### 1. **Clone the repository:**

git clone https://github.com/your_username/youtube-video-summarizer.git
cd youtube-video-summarizer

    
### 2\. **Install dependencies:**

    pip install -r requirements.txt
    

### 3\. **Required API Keys:**

*   You will need an **Google Gemini API key** for text summarization and answering questions. Enter your API key in the sidebar after running the app.

* * *

**Usage**
---------

### 1\. **Run the Streamlit app:**

    streamlit run app.py
    

### 2\. **Input the Video URL:**

*   Enter the **YouTube video URL** in the sidebar to fetch the transcript.

### 3\. **Process Video:**

*   Click the **"Process Video"** button to extract the transcript, chunk the text, and embed it using pre-trained models.

### 4\. **Generate Summary:**

*   Use the **"Generate Summary"** button to get a summary of the video transcript.

### 5\. **Ask a Question:**

*   Enter a question in the **"Ask a Question"** section and select a re-ranking method (No Re-Ranking, Cross-Encoder, BM25).
*   The system will rank the document chunks based on relevance and generate an answer using **GPT-based models**.

* * *

**Project Structure**
---------------------

    ├── app.py                    # Main Streamlit app file
    ├── requirements.txt          # Python dependencies
    ├── output.txt                # Output file for transcript text (optional)
    └── README.md                 # Project documentation
    

* * *

**Models**
----------

The following models are used in the application:

1.  **Sentence Transformers (Paraphrase-XLM-R-Multilingual-v1):**  
    Used to generate embeddings for video text chunks.
    
2.  **Cross-Encoder (MS-MARCO-MiniLM-L-6-v2):**  
    Used for scoring query-document pairs to rank relevant chunks.
    
3.  **BM25:**  
    A traditional ranking method for document retrieval used for comparing query and document relevance.
    
4.  **Google Gemini Models:**  
    Used for generating summaries and answering questions based on the video content.
    

* * *

**Results**
-----------

*   **Performance Metrics:**
    
    *   High-quality video summary generation.
    *   Accurate answers based on document ranking methods.

    

* * *

**Challenges and Improvements**
-------------------------------

### **Challenges:**

*   **Transcript Quality:**  
    The quality of the transcript depends on the video’s captioning system. Inaccuracies in transcription can affect the quality of the summary and answers.
    
*   **Question Relevance:**  
    Some queries may not get perfect answers, especially if the transcript doesn't fully cover the context.
    

### **Future Improvements:**

*   **Better Ranking Methods:**  
    Improve ranking algorithms to better understand and rank context-specific answers.
    
*   **Support for Multiple Languages:**  
    Extend support for generating summaries and answering questions in multiple languages.
