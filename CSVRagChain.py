#Imports
import pandas as pd
import streamlit as st
import os
import openai
import plotly.express as px
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.evaluation.qa import QAEvalChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationTokenBufferMemory, ConversationSummaryBufferMemory
from langchain.chains import RetrievalQA, LLMChain
from dotenv import load_dotenv
from langchain.chains.combine_documents.stuff import StuffDocumentsChain


#Configurations
load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
FAISS_INDEX_PATH = "faiss_index"
CSV_FILE = "sales_data.csv"
OPENAI_MODEL = "gpt-4o-mini"
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0) 

#Load CSV using Panda's 
df = pd.read_csv("sales_data.csv")

# Sort by Product, Region,Date 
df = df.sort_values(by=["Product", "Region", "Date"], ascending=[True, True, True])

# 2. Convert sorted rows into strings
rows_as_text = df.astype(str).apply(lambda row: ", ".join(row.values), axis=1).tolist()

# 3. Chunk rows before embeddings
def chunk_rows(rows, chunk_size=20):
    """Group rows into text chunks for embedding"""
    for i in range(0, len(rows), chunk_size):
        yield "\n".join(rows[i:i+chunk_size])
        
headers = ", ".join(df.columns)
chunks = [headers + "\n" + chunk for chunk in chunk_rows(rows_as_text, chunk_size=20)]

# Wrap chunks into Document objects
docs = [Document(page_content=chunk) for chunk in chunks]

# Initialize embeddings
openai_embed = OpenAIEmbeddings()


#Check if VectorStore is present if not create one on local. 
if os.path.exists(FAISS_INDEX_PATH):
    print("✅ Found existing FAISS index. Loading...")
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, openai_embed, allow_dangerous_deserialization=True)
else:
    print("⚡ No FAISS index found. Building a new one...")
        
    # Build FAISS index
    vectorstore = FAISS.from_documents(docs, openai_embed)
        
    # Save FAISS index for reuse
    vectorstore.save_local(FAISS_INDEX_PATH)
    print("✅ Vector store created with grouped row chunks")

# --- Custom Retriever ---
class StatsRetriever:
    def __init__(self, vectorstore, dataframe):
        self.vectorstore = vectorstore
        self.df = dataframe

# k represnt numbers of chunks the retriver will pull. 
    
    def retrieve(self, query: str, k: int = 200):
        """
        Hybrid retriever:
        1. Use vectorstore for semantic similarity search.
        2. Use pandas for quick stats if query matches known metrics.
        Always returns (semantic_hits).
        """
        semantic_hits = self.vectorstore.similarity_search(query, k=k)
        return semantic_hits     

retriever = StatsRetriever(vectorstore, df)

# --- Prompt Engineering ---
prompt_template = """
You are a data assistant. Answer the user’s question using both the retrieved context and statistics.

Question: {query}


Guidelines:
1. If statistics are available, prioritize them over context.
2. If statistics contradict the context, present both, highlight the discrepancy, and suggest possible reasons.
3. If no statistics are provided, answer using only the context.
4. Keep answers clear and informative, aiming for ~200 words maximum.
5. If query is not relevent to context then sarcasticly tell user to ask relevent question about context only. 
"""

prompt = PromptTemplate(
    input_variables=["query"],
    template=prompt_template,
)

memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=500)

# chain using memory
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# AnalyticalAssitant Class for using LLM to pull details. 
class AnalyticsAssistant:
    def __init__(self, retriever, chain):
        self.retriever = retriever 
        self.chain = chain

    def ask(self, user_query: str):
        # Retrieve from vectorstore + pandas stats 
        semantic_hits = self.retriever.retrieve(user_query)

        # Prepare context and stats
        context_text = "\n".join([hit.page_content for hit in semantic_hits])

        # Pass to LLM chain
        response = self.chain.run(
            query=f"Question: {user_query}\nContext:\n{context_text}"
        )

        return { 
            "user_query": user_query, 
            "semantic_hits": semantic_hits, 
            "response": response 
            }

assistant = AnalyticsAssistant(retriever, chain)

#Evaluate Model using QAEval
eval_chain = QAEvalChain.from_llm(llm)

examples = [
    {
        "query": "What is total sales for Widget A in year 2022 in north region?",
        "answer": "1000"
    },
    {
        "query": "What is total sales by males in year 2025 in west region'?",
        "answer": "1000"
    }
]

predictions = [
    {"result": "1000"},  # correct
    {"result": "200"}  # incorrect
]

# Run evaluation
graded_outputs = eval_chain.evaluate(
    examples,
    predictions,
    question_key="query",
    prediction_key="result",
    answer_key="answer"
)

# Streamlit UI 

ANALYSIS_TYPE = ["Sales trend over time", "Product performance comparisons", "Regional analysis", "Customer demographics and segmentation"]
PRODUCTS = ["All Products"] + df["Product"].unique().tolist()
REGIONS = ["All Regions"] + df["Region"].unique().tolist() 
GENDER = ["Both Genders"] + df["Customer_Gender"].unique().tolist()

def run_streamlit_app():
    
    if "history" not in st.session_state:
        st.session_state.history = []

    st.title("InsightForge - Business Intelligence Assistant")
    user_input = st.text_input("Enter a question above and press Send.")

    df = pd.read_csv("sales_data.csv")


    with st.sidebar:
        st.header("VISUAL REPRESENTATIONS : ")
        graph = st.selectbox("Analitical Graphs", options=ANALYSIS_TYPE, index=0)

        st.header("FILTERS : ")
        product = st.selectbox("Product", options=PRODUCTS, index=0)
        region = st.selectbox("Region", options=REGIONS, index=0)
        gender = st.selectbox("Gender", options=GENDER, index=0)

        plot_graph = st.button("Plot Graph")

    if st.button("Send"):
        if user_input:
            result = assistant.ask(user_input)
            st.write(result["response"])
            st.session_state.history.append({"user": user_input, "assistant": result["response"]})
        else:
            st.warning("Please enter a question first.")

    if plot_graph:
        # Apply filter if not ALL
        if product != "All Products":
            df = df[df["Product"] == product]
        if region != "All Regions":
            df = df[df["Region"] == region]
        if gender != "Both Genders":
            df = df[df["Customer_Gender"] == gender]

        if graph == "Sales trend over time":
            df["Date"] = pd.to_datetime(df["Date"])
            df["YearMonthStr"] = df["Date"].dt.strftime("%Y-%m")
            df = df.groupby("YearMonthStr")["Sales"].sum().reset_index()
            st.subheader("📈 Sales Trend Over Time")
            st.line_chart(df.set_index("YearMonthStr")["Sales"])

        if graph == "Product performance comparisons":
            df["Date"] = pd.to_datetime(df["Date"])
            df["Year"] = df["Date"].dt.strftime("%Y")
            df = df.groupby(["Year","Product"])["Sales"].sum().reset_index()
            # Pivot data to have Products as columns
            df_pivot = df.pivot(index="Year", columns="Product", values="Sales").fillna(0)
            st.subheader("📈 Product performance comparisons")
            st.line_chart(df_pivot)

        if graph == "Regional analysis":
            df["Date"] = pd.to_datetime(df["Date"])
            df["Year"] = df["Date"].dt.strftime("%Y")
            df = df.groupby(["Year","Region"])["Sales"].sum().reset_index()
            # Pivot data to have Region as columns
            df_pivot = df.pivot(index="Year", columns="Region", values="Sales").fillna(0)
            st.subheader("📈 Regional analysis")
            st.bar_chart(df_pivot)

        if graph == "Customer demographics and segmentation":
            bins = [0, 20, 30, 40, 50, 60, 100]  # Age ranges
            labels = ["0-20", "21-30", "31-40", "41-50", "51-60", "60+"]  # Labels for groups
            df["AgeGroup"] = pd.cut(df["Customer_Age"], bins=bins, labels=labels, right=True)
            df= df.groupby("AgeGroup")["Sales"].sum().reset_index()
            st.subheader("📈 Customer demographics and segmentation")
            st.line_chart(df.set_index("AgeGroup")["Sales"])
            
if __name__ == "__main__":
    run_streamlit_app()