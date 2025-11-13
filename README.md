# 🤖 RAG-Chatbot-Intelligent-Investor

This project is a **Retrieval-Augmented Generation (RAG) based chatbot** built using pretrained models and Chroma vector DB. The system is designed to read and understand complex text (in our case, *"The Intelligent Investor by Benjamin Graham"*) and provide meaningful answers, summaries, and context-aware responses.

---

## 📚 What It Does

* Processes and stores book text into the **Chroma vector DB** for efficient retrieval.
* Uses **RAG** to combine search with generation, making answers more accurate.
* Supports multiple models for different tasks:

  * **TinyLlama (HuggingFace):** Query-based responses.
  * **BART (HuggingFace):** Summarization of retrieved content.
  * **Phi-3 (Ollama with LangChain libraries):** Context-aware responses.

---

## 🔍 How It Works

1. **Data Preprocessing (main.py)**

   * Reads the book PDF.
   * Splits and cleans text.
   * Embeds content and stores it in a **Chroma vector DB**.
   * Defines, tests, and evaluates models using cosine similarity.

2. **Backend Logic (GUI_class.py & GUI_main.py)**

   * `GUI_class.py`: Defines model classes and loads the saved vector DB.
   * `GUI_main.py`: Handles user queries, runs retrieval + model pipelines, and serves results through Flask.

3. **Frontend (GUI.html)**

   * Simple, browser-based interface to interact with the chatbot.
   * Connects to Flask backend to display answers and summaries.

---

## 🌟 Key Features

* **Vector Search with Chroma** – fast and relevant document retrieval.
* **Multiple Models Integrated** – response generation, summarization, and context understanding.
* **Flask-based GUI** – easy to use in the browser.
* **Evaluation with Cosine Similarity** – ensures better accuracy and relevance.

---

## 📂 Project Structure

```bash
├── GUI.html          # Frontend interface
├── GUI_class.py      # Model classes & vector DB loading
├── GUI_main.py       # Flask server – handles queries & serves responses
├── main.py           # Preprocessing, DB creation, model definition & evaluation
└── README.md         # Project documentation
```

---

## 👥 Team

**Author:**

* Abdulmoiz

**Contributors:**

* Wasif Mehboob
* Ahsan Waseem

---

## 🚀 Running the Project

1. **Install Dependencies**

   ```bash
   pip install flask langchain chromadb transformers sentence-transformers
   ```

2. **Preprocess & Build DB**

   ```bash
   python main.py
   ```

3. **Start Flask Server**

   ```bash
   python GUI_main.py
   ```

4. **Open in Browser**
   Navigate to: [http://127.0.0.1:5000](http://127.0.0.1:5000) or any other default port.

---
