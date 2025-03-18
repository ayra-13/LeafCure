import os
import sqlite3
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

# Load API keys
load_dotenv()

# Initialize embeddings & vector store
embeddings = CohereEmbeddings(
    model="embed-english-light-v3.0",
    cohere_api_key=os.getenv("COHERE_API_KEY")
)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

class ChatAgent:
    def __init__(self, session_id: str):
        self.llm = ChatGroq(model_name="mixtral-8x7b-32768", api_key=os.getenv("GROK_API_KEY"))
        self.session_id = session_id

    def retrieve_context(self, query: str):
        """Retrieve relevant text chunks from ChromaDB with score filtering."""
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)

        # Filter out irrelevant results (only keep scores ≥ 0.7)
        filtered_docs = [doc for doc, score in docs_with_scores if score >= 0.7]

        if not filtered_docs:
            return "I could not find relevant information in the database."

        # Combine document text
        context = "\n".join([doc.page_content for doc in filtered_docs])
        # print(f"🔍 Retrieved Context: {context}")  # Debugging retrieval output

        return context

    def get_last_5_chats(self):
        """Fetch last 5 user-bot exchanges for maintaining context."""
        conn = sqlite3.connect("chat_history.db")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT user_input, bot_response FROM chats WHERE session_id = ? ORDER BY timestamp DESC LIMIT 5",
            (self.session_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        # Format chat history as a readable string
        chat_history = "\n".join([f"User: {row[0]}\nBot: {row[1]}" for row in reversed(rows)])  # Reverse to maintain order
        return chat_history

    def chat(self, user_input: str):
        """Handles user input, retrieves relevant context, and generates a response."""
        
        context = self.retrieve_context(user_input)

        # If retrieval fails, return an explicit "no information" response
        if not context.strip() or "I could not find relevant information" in context:
            return "I do not have information on this topic."



        chat_history = self.get_last_5_chats()

        prompt = ChatPromptTemplate.from_template(
            """You are an intelligent AI assistant. **Follow these rules**:
            
            1️⃣ If the user input is **greetings or casual chat**, keep responses **brief and natural**.
            2️⃣ If the query is **technical or knowledge-based**, provide an **accurate response using retrieved context**.
            3️⃣ **Do NOT generate answers outside of retrieved context**.
            4️⃣ If **no relevant information is found**, ask for **clarification instead of making up answers**.

            ---
            🔹 **Chat History (Last 5 Messages):** 
            {history}

            🔹 **User Input:** {input}
            🔹 **Retrieved Context (Do not include unrelated details):** {context}

            ❗ **If the context is empty or unclear, ask for clarification instead of guessing.**
            ❗ **Answer STRICTLY using the retrieved context from the database.**
            ❗ **If no relevant context is retrieved, say: "I do not have information on this topic." DO NOT suggest related topics from the database. DO NOT redirect the user to another question.**

            """
        )

        chain = (
            RunnablePassthrough.assign(
                history=lambda x: self.get_last_5_chats(),
                context=lambda x: self.retrieve_context(x["input"])
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke({"input": user_input, "context": context, "history": chat_history})
        self._store_conversation(user_input, response)
        return response

    def _store_conversation(self, user_input: str, bot_response: str):
        """Stores chat history in SQLite."""
        conn = sqlite3.connect("chat_history.db")
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chats (session_id, user_input, bot_response) VALUES (?, ?, ?)",
            (self.session_id, user_input, bot_response),
        )
        conn.commit()
        conn.close()
