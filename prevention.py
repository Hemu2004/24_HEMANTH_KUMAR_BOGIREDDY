import pandas as pd
import google.genai as genai
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st

df_precaution = pd.read_csv("Disease precaution.csv")

# RAG Setup
@st.cache_resource
def setup_rag():
    try:
        # Load embedding model
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Prepare documents for embedding
        documents = []
        for _, row in df_precaution.iterrows():
            disease = row['Disease']
            precautions = [str(p) for p in row[1:] if pd.notna(p) and str(p).strip()]
            doc = f"Disease: {disease}. Precautions: {'; '.join(precautions)}"
            documents.append(doc)

        # Create embeddings
        embeddings = model.encode(documents)

        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))

        return model, index, documents
    except Exception as e:
        print(f"Error setting up RAG: {e}")
        return None, None, None

model, index, documents = setup_rag()

def retrieve_relevant_precautions(query, top_k=3):
    """
    Retrieve top-k relevant precautions from the vector database based on the query.
    """
    if model is None or index is None or documents is None:
        return []

    try:
        # Encode the query
        query_embedding = model.encode([query])

        # Search the FAISS index
        distances, indices = index.search(query_embedding.astype('float32'), top_k)

        # Retrieve relevant documents
        relevant_docs = [documents[i] for i in indices[0] if i < len(documents)]
        return relevant_docs
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return []

def get_precautions(disease):
    row = df_precaution[df_precaution["Disease"] == disease]

    if row.empty:
        return []

    precautions = row.iloc[0, 1:].dropna().tolist()
    return precautions

def generate_llm_precautions(symptoms, diseases, common_diseases=None):
    """
    Generate user-friendly precautions using Google AI Studio based on symptoms, predicted diseases, and common diseases.
    Augments LLM generation with retrieved context from RAG.
    """
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "API key not found. Please set GOOGLE_API_KEY environment variable."
        client = genai.Client(api_key=api_key)
        model = "gemini-2.5-flash"

        all_diseases = [d for d, _ in diseases] + (common_diseases or [])

        # Retrieve relevant precautions using RAG
        query = f"Symptoms: {', '.join(symptoms)}. Diseases: {', '.join(all_diseases)}"
        relevant_precautions = retrieve_relevant_precautions(query, top_k=5)
        context = "\n".join(relevant_precautions) if relevant_precautions else "No specific precautions found in database."

        prompt = f"""
        Based on the following symptoms: {', '.join(symptoms)}
        And possible diseases: {', '.join(all_diseases)}

        Relevant context from precaution database:
        {context}

        Provide user-friendly, clear, and actionable precautions or advice in simple language.
        Focus on prevention, management, and when to seek medical help.
        Keep it concise, empathetic, and non-alarming.
        Structure the response as a list of bullet points.
        Use the provided context to inform your advice where relevant.
        """

        response = client.models.generate_content(
            model=model,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Unable to generate precautions at this time. Error: {str(e)}. Please consult a healthcare professional."
