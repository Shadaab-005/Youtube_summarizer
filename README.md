# YouTube Video Summarizer & Q&A

## Overview
This web application allows users to extract transcripts from YouTube videos, generate summaries, and perform question-answering using different ranking methods. The application leverages NLP and deep learning techniques to provide accurate and context-aware responses.

## Features
- Extract and process YouTube video transcripts.
- Generate AI-powered summaries using OpenAI's GPT models.
- Ask questions based on the video content with three different ranking methods:
  - No Re-Ranking (cosine similarity-based ranking)
  - Cross-Encoder-based Re-Ranking
  - BM25 Algorithm
- Interactive Streamlit UI for ease of use.

## Technologies Used
- **Streamlit**: Web framework for UI development
- **YouTubeTranscriptApi**: Fetching video transcripts
- **Sentence-Transformers**: Embedding-based similarity search
- **LangChain**: Document chunking and text processing
- **OpenAI GPT Models**: Generating summaries and responses
- **BM25**: Traditional information retrieval method

## Installation
### Prerequisites
Ensure you have Python 3.8+ installed.

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/youtube-summarizer.git
   cd youtube-summarizer
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Enter the YouTube video URL in the sidebar.
2. (Optional) Enter your OpenAI API key for enhanced summary and Q&A capabilities.
3. Click **Process Video** to extract and process the transcript.
4. Generate a summary or ask questions based on the transcript.
5. Select a ranking method and view ranked document excerpts.
6. If using OpenAI API, generate AI-powered responses.

## API Keys
- An OpenAI API key is required for generating summaries and LLM-based responses.
- Enter the key in the sidebar to enable these features.

## Future Enhancements
- Support for multiple languages.
- Improved transcript processing and noise handling.
- Integration with more LLM models for enhanced responses.



