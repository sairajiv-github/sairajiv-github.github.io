# Spiritual Q&A with Bhagavad Gita and Ramayana

A Gemini-powered spiritual assistant that answers questions based on the Bhagavad Gita, Ramayana, or general spiritual wisdom using RAG (Retrieval-Augmented Generation) and multi-agent systems.

## How to Run:

1. **Install requirements:**
   ```bash
   pip install google-generativeai PyMuPDF scikit-learn tqdm crewai fitz
   Prepare your PDFs:

Place gita.pdf and ramayana.pdf in the root directory

Or edit the filenames in the code if using different texts

Set up API key:

Replace "AIzaSyCCx7fMEGjfCqMxGvBxKONiOGJ3SSTIN9Q" with your actual Gemini API key

Run the script:

bash
python spiritual_qa.py
Use the application:

*The script will prompt you to ask a spiritual question

*It will automatically:

*Classify your question (Gita/Ramayana/General)

*Retrieve relevant passages

*Generate an answer

*Refine it for spiritual warmth

Agent Roles:
*The system uses these specialized agents:

*Spiritual Classifier: Routes questions to the right expert

*Gita Scholar: Answers from Bhagavad Gita wisdom

*Ramayana Narrator: Answers from Ramayana stories

*Philosophy Guru: Handles general spiritual questions

*Friendly Guru: Refines answers with poetic warmth

Features:
*PDF text extraction and intelligent chunking

*Embedding generation with caching

*Cosine similarity for relevant passage retrieval

*Multi-agent workflow for specialized answers

*Response refinement for spiritual tone
