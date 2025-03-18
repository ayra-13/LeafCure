# 🌾 Rice Disease RAG Chatbot

## 📌 Overview
This chatbot is designed to help users retrieve **accurate, document-based** information about rice diseases. It uses **Retrieval-Augmented Generation (RAG)** to ensure responses are sourced from **uploaded PDFs**.  

## 🚀 Features
✅ Uses **LangChain** for retrieval and LLM-based responses  
✅ Stores embeddings in **ChromaDB** for fast retrieval  
✅ **No hallucinations** – answers only from available documents  
✅ **Maintains chat history** (last 5 interactions)  
✅ Built with **Streamlit** for an easy-to-use UI  

---

## 📂 Folder Structure
📂 rice_disease_chatbot/ 

│── 📂 data/ # Folder for your rice disease PDFs 

│── 📂 chroma_db/ # Stores vector embeddings (auto-created) 

│── app.py # Runs the Streamlit chatbot UI 

│── bot.py # Handles retrieval and response generation 

│── chunking.py # Processes PDFs and stores embeddings 

│── db.py # Sets up SQLite database for chat history 

│── chat_history.db # Auto-created SQLite DB 

│── requirements.txt # Python dependencies 

│── .env # Stores API keys 

│── README.md # Documentation (this file)


## 🔧 **Setup & Installation**
### **1️⃣ Install Dependencies**
First, install required packages:  
```bash
pip install -r requirements.txt
```

2️⃣ Add Your API Keys
Create a .env file in the project folder and add:
```bash
COHERE_API_KEY=your_cohere_api_key
GROK_API_KEY=your_groq_api_key
```
Replace your_cohere_api_key and your_groq_api_key with real API keys.


3️⃣ Process PDFs and Store Embeddings
Move your rice disease PDFs into rice_disease_pdfs/ and run:
```bash
python chunking.py
```
✅ This will store the embeddings in ChromaDB.


4️⃣ Initialize the Chat History Database
Run db.py to create chat_history.db:

```bash
python db.py
```
✅ This sets up SQLite database for storing chat history.


5️⃣ Start the Chatbot
Run:

```bash
streamlit run app.py
```
✅ Opens the chatbot in your web browser!


### 🛠️ How It Works
1. User asks a question.
2. Bot retrieves relevant text from PDFs using ChromaDB.
3. LLM generates an answer using only retrieved context.
4. Bot saves the conversation history.
5. If no relevant information is found, the bot says: "I do not have information on this topic."


### 🔬 Testing the Chatbot
Test	Expected Response

🔹 "What is bacterial leaf blight?"	✅ Retrieves relevant info from PDFs

🔹 "What is the role of AI in curing rice diseases?"	✅ "I do not have information on this topic."

🔹 "Tell me about Mars colonization."	✅ "I do not have information on this topic."

🔹 "What is quantum leaf blight?"	✅ "I do not have information on this topic."


### 💡 Future Improvements
🔹 Add multi-user chat history

🔹 Improve UI/UX for mobile devices

🔹 Deploy online using Hugging Face Spaces or Streamlit Cloud


### 🤝 Contributions & Support
Feel free to report issues or suggest features! 🚀
If you need help, reach out via GitHub Issues.



---

✅ **Now, just save this as `README.md` in your project folder!**  
Let me know if you need any modifications. 🚀🔥