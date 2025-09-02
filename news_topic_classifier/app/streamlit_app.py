import os
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from utils.labels import LABELS

st.set_page_config(page_title="News Topic Classifier", page_icon="📰")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "news_topic_classifier")

@st.cache_resource
def load():
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    return model, tokenizer

model, tokenizer = load()

st.title("📰 News Topic Classifier (BERT)")
st.write("Enter a news headline to get the predicted category and confidence scores.")

text = st.text_input("Headline", placeholder="e.g., Apple unveils new iPhone at annual event")

if st.button("Classify") and text:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]

    pred_idx = int(torch.argmax(probs).item())
    st.subheader(f"Predicted: {LABELS[pred_idx]}")
    st.write("Confidence scores:")
    for i, label in enumerate(LABELS):
        st.write(f"- {label}: {probs[i].item():.4f}")
