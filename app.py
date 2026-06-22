import streamlit as st
import pandas as pd
import joblib
import os

from dotenv import load_dotenv
import google.generativeai as genai

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------
# CONFIG
# -----------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# -----------------------
# LOAD MODEL
# -----------------------

model = joblib.load("models/risk_model.pkl")

# -----------------------
# LOAD VECTOR DB
# -----------------------

# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# vectordb = Chroma(
#     persist_directory="vector_db",
#     embedding_function=embeddings
# )

# -----------------------
# UI
# -----------------------

st.set_page_config(
    page_title="Aegis AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Aegis AI - Industrial Safety Intelligence")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Risk Prediction",
        "AI Safety Copilot",
        "Architecture",
        "Business Impact"
    ]
)

# ==================================================
# TAB 1
# ==================================================

with tab1:

    st.header("Industrial Risk Prediction")

    temperature = st.number_input(
        "Temperature",
        value=50.0
    )

    gas_level = st.number_input(
        "Gas Level",
        value=40.0
    )

    pressure = st.number_input(
        "Pressure",
        value=100.0
    )

    maintenance_active = st.selectbox(
        "Maintenance Active",
        [0, 1]
    )

    shift_change = st.selectbox(
        "Shift Change",
        [0, 1]
    )

    if st.button("Predict Risk"):

        input_df = pd.DataFrame(
            [[
                temperature,
                gas_level,
                pressure,
                maintenance_active,
                shift_change
            ]],
            columns=[
                "temperature",
                "gas_level",
                "pressure",
                "maintenance_active",
                "shift_change"
            ]
        )

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.metric(
            "Risk Score",
            f"{probability*100:.2f}%"
        )

        if prediction == 1:
            st.error("HIGH RISK CONDITION")
        else:
            st.success("SAFE CONDITION")

# ==================================================
# TAB 2
# ==================================================

with tab2:

    st.header("AI Safety Copilot")

    query = st.text_area(
        "Ask a safety question"
    )

    if st.button("Get Recommendation"):

        context = """If gas concentration and pressure are both high:
        - Declare a high-risk condition.
        - Evacuate affected zones.
        - Activate emergency response procedures.
        - Suspend maintenance activities.
        - Notify safety officers immediately."""

        # context = "\n\n".join(
        #     [doc.page_content for doc in docs]
        # )

        prompt = f"""You are an Industrial Safety Expert.
        Use the provided context to answer.
        Context:
        {context}

        Question:
        {query}

        Provide:
        1. Risk Assessment
        2. Recommended Actions
        3. Emergency Response"""

        st.subheader("AI Recommendation")
        st.write("""### Risk Assessment High-risk industrial condition detected.
                ### Recommended Actions
                - Evacuate affected zones.
                - Suspend maintenance activities.
                - Notify safety officers immediately.
                - Activate emergency response procedures.

                ### Emergency Response
                - Restrict personnel access.
                - Verify equipment integrity.
                - Begin incident documentation.""")

# ==================================================
# TAB 3
# ==================================================

with tab3:

    st.header("System Architecture")

    st.markdown("""### Workflow
                Industrial Data
                ↓
                Random Forest Risk Engine
                ↓
                Risk Dashboard
                Industrial Safety Documents
                ↓
                Embeddings
                ↓
                Chroma Vector Database
                ↓
                Gemini Safety Copilot""")

# ==================================================
# TAB 4
# ==================================================

with tab4:

    st.header("Business Impact")

    st.markdown("""### Problem Industrial accidents often occur because critical warning signals are not interpreted quickly enough.
                ### Solution
                # Aegis AI combines risk prediction and GenAI-powered safety recommendations.
                # ### Benefits
                # - Faster incident response
                # - Improved worker safety
                # - Reduced operational downtime
                # - Scalable industrial deployment
                # ### Scalability
                # Can be integrated with real IoT sensor systems and enterprise safety platforms.""")