import streamlit as st
import pandas as pd
from model import predict_top3, symptom_columns
from prevention import get_precautions

st.set_page_config(page_title="AI Health Assistant")

st.title("🩺 AI Health Assistant")
st.write("Describe your symptoms in natural language")

# Session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Type your symptoms here...")

if user_input:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("assistant"):
        st.markdown("Analyzing your symptoms...")

    # Simple symptom extraction
    detected_symptoms = [
        s for s in symptom_columns if s.replace("_", " ") in user_input.lower()
    ]

    # Create binary input
    input_data = {s: 0 for s in symptom_columns}
    for s in detected_symptoms:
        input_data[s] = 1

    input_df = pd.DataFrame([input_data])

    # Prediction
    top3 = predict_top3(input_df)

    # AI Response
    response = "### 🔍 Possible Conditions\n"
    for d, s in top3:
        response += f"- **{d}** ({round(s*100,1)}%)\n"

    response += "\n### 🛡️ Prevention Tips\n"
    for d, _ in top3:
        response += f"**{d}:**\n"
        for p in get_precautions(d):
            response += f"- {p}\n"

    response += "\n⚠️ *This is not a medical diagnosis. Consult a doctor.*"

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    st.rerun()
