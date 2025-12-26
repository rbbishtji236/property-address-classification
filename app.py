import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re
import os
from config import *

st.set_page_config(
    page_title="Property Address Classifier",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 20px 0;
    }
    .category-label {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .confidence-label {
        font-size: 1.5rem;
        color: #27ae60;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_tokenizer():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s,.\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict_address(address, model, tokenizer):
    
    text = clean_text(address)
    
    encoding = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]
        prediction = torch.argmax(probs).item()
    
    category = ID2LABEL[prediction]
    confidence = probs[prediction].item()
    all_probs = probs.cpu().numpy()
    
    return category, confidence, all_probs

def main():
    
    st.markdown('<p class="main-header"> Property Address Classifier</p>', unsafe_allow_html=True)
    
    tokenizer, model = load_model_and_tokenizer()
    
    if model is None:
        st.stop()
    
    st.markdown("---")
    
    st.subheader("Enter Property Address")
    
    default_text = st.session_state.get('sample_address', '')
    
    address_input = st.text_area(
        "Typethe property address below:",
        value=default_text,
        height=100,
        placeholder="e.g., Flat-697, Floor-Seventh, Pavithra Olympus, Bengaluru 573201 Karnataka"
    )
    
    predict_button = st.button("Classify Address", type="primary", use_container_width=True)
    
    if predict_button and address_input.strip():
        
        with st.spinner("Analyzing address..."):
            
            category, confidence, probs = predict_address(address_input, model, tokenizer)
            
            if 'sample_address' in st.session_state:
                del st.session_state.sample_address
        
        st.markdown("---")
        st.subheader("Prediction Results")
        
        st.markdown(
            f"""
            <div class="prediction-box">
                <p style="font-size: 1rem; color: #7f8c8d; margin-bottom: 5px;">Predicted Category:</p>
                <p class="category-label">{category.upper()}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Probability chart
        st.markdown("---")
        st.subheader("All Category Probabilities")
        
        st.markdown("#### Detailed Probabilities")
        prob_df = pd.DataFrame({
            'Category': CATEGORIES,
            'Probability': [f"{p:.2%}" for p in probs]
        })
        prob_df = prob_df.sort_values('Probability', ascending=False).reset_index(drop=True)
        prob_df.index = prob_df.index + 1
        st.dataframe(prob_df, use_container_width=True)
        
    elif predict_button and not address_input.strip():
        st.warning("Please enter a property address to classify.")
    

if __name__ == "__main__":
    main()