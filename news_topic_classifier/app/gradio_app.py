import os
import gradio as gr
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from utils.labels import LABELS

MODEL_DIR = r"C:\Users\AAFIA\Desktop\news_topic_classifier\models\news_topic_classifier"


model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)

def classify(text):
    if not text or text.strip() == "":
        return {label: 0.0 for label in LABELS}
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]
    return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=2, placeholder="Enter a news headline..."),
    outputs=gr.Label(num_top_classes=4),
    title="News Topic Classifier (BERT)",
    description="Predicts one of: World, Sports, Business, Sci/Tech"
)

if __name__ == "__main__":
    demo.launch()
