A spiritual Agentic AI powered by Google's Gemini and formed by crewAI, answering questions about the Bhagavad Gita and Ramayana using Retrieval-Augmented Generation (RAG). It extracts wisdom from uploaded PDFs, retrieves relevant passages, and generates poetic responses with multi-agent workflows (classification → retrieval → generation → refinement).

Cell 1 - This cell uploads PDF files providing source documents for RAG search.(saves gita.pdf and ramayana.pdf in runtime)
Cell 2 - This cell installs the dependencies like PyMuPDF(extracts text from PDFs), google-generativeai(Accesses Gemini for embeddings/LLM) and scikit-learn(computes cosine similarity for retrieval)
Cell 3 - Imports the required libraries for embeddings,text processings and similarity search
Cell 4 - Sets up Gemini API for embeddings (embedding-001) and generation (gemini-1.5-flash).
Cell 5 - Implements RAG : get_top_chunks() finds relevant text using cosine similarity(Retrieval), the chunks are provided to prompts(Augmentation) and ask_gemini() produces answers based on retrieved context(generation)
Cell 6 - Converts PDFs into chunks and embeddings
Cell 7 - forms and assigns roles to the crew agents
Cell 8 - provides with the Task definitions for the crew agents
Cell 9 - Implements the cells in sequential manner
Cell 10 - User Interaction
