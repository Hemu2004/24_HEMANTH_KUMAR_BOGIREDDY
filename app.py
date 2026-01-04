import streamlit as st
import pandas as pd
from model import predict_top3, symptom_columns
from prevention import get_precautions, generate_llm_precautions
import re
import os

st.set_page_config(page_title="AI Health Assistant")

st.title("🩺 AI Health Assistant")
st.write("Describe your symptoms in natural language or select from the list below")

# Sidebar for symptom selection
st.sidebar.header("Select Symptoms")
selected_symptoms = []
for symptom in symptom_columns:
    clean_symptom = symptom.replace("_", " ").title()
    if st.sidebar.checkbox(clean_symptom):
        selected_symptoms.append(symptom)

# Session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Function to generate response
def generate_response(detected_symptoms):
    # Create binary input
    input_data = {s: 0 for s in symptom_columns}
    for s in detected_symptoms:
        input_data[s] = 1

    input_df = pd.DataFrame([input_data])

    # Prediction
    top3 = predict_top3(input_df)

    # Common diseases to include
    common_diseases = ["Common Cold", "Allergy", "Headache"]

    # AI Response
    response = "### 🔍 Possible Conditions\n"
    for d, s in top3:
        response += f"- **{d}** ({round(s*100,1)}%)\n"

    response += "\n### 🛡️ Prevention Tips\n"
    # Use LLM for user-friendly precautions based on symptoms, top diseases, and common diseases
    llm_precautions = generate_llm_precautions(detected_symptoms, top3, common_diseases)
    response += llm_precautions

    response += "\n⚠️ *This is not a medical diagnosis. Consult a doctor for proper medical advice.*"
    return response

# User input
user_input = st.chat_input("Type your symptoms here...")

if user_input:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("assistant"):
        st.markdown("Analyzing your symptoms...")

    # Improved symptom extraction
    detected_symptoms = []
    user_text = user_input.lower()

    for symptom in symptom_columns:
        # Clean symptom name for matching
        clean_symptom = symptom.replace("_", " ").strip()

        # Check for exact match
        if clean_symptom in user_text:
            detected_symptoms.append(symptom)
            continue

        # Check for partial matches (symptom words)
        symptom_words = clean_symptom.split()
        if len(symptom_words) > 1:
            # For multi-word symptoms, check if all words are present
            if all(word in user_text for word in symptom_words):
                detected_symptoms.append(symptom)
                continue

        # Check for fuzzy matches using regex (allowing for typos)
        # Replace spaces with flexible pattern
        pattern = r'\b' + re.escape(clean_symptom.replace(" ", r"\s+")) + r'\b'
        if re.search(pattern, user_text, re.IGNORECASE):
            detected_symptoms.append(symptom)

    # Add selected symptoms from checkboxes
    detected_symptoms.extend(selected_symptoms)

    # Remove duplicates
    detected_symptoms = list(set(detected_symptoms))

    response = generate_response(detected_symptoms)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    st.rerun()

# Check button for selected symptoms
if st.sidebar.button("Check"):
    if selected_symptoms:
        st.sidebar.success("Analyzing selected symptoms...")
        response = generate_response(selected_symptoms)
        st.markdown(response)
    else:
        st.sidebar.warning("Please select at least one symptom.")
