# 📰 News Topic Classifier (AG News + BERT)

## 📖 Description  
**✨ Developed as part of AI/ML Engineering Internship – DevelopersHub Corporation**
This project is a **News Topic Classifier** built using **BERT** and the **AG News dataset**.  
It classifies news articles into **four categories**:  
- 🌍 World  
- 🏅 Sports  
- 💼 Business  
- 🔬 Sci/Tech  

The pipeline covers **data preprocessing, tokenization, model training, evaluation**, and **inference**. A Gradio app is also included for real-time predictions.

---

## ⚙️ Installation  

1. Clone this repository:
```bash
git clone https://github.com/aafia1/news_topic_classifier.git
cd news_topic_classifier
```

2. Create and activate a virtual environment (recommended):
```bash
conda create -n agnews python=3.10 -y
conda activate agnews
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Training the Model
To train the model on AG News dataset:
```bash
python train.py
```
This will save the fine-tuned model inside `models/news_topic_classifier`.

---

## ⚡ Quick Prediction Example  
Once the model is trained (or you download a pretrained one), you can test it on custom headlines:  

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer
model_path = "./models/news_topic_classifier"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

labels = ["World", "Sports", "Business", "Sci/Tech"]

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=1)
    return labels[preds.item()]

# Example
headline = "NASA launches new spacecraft to study the Sun"
print("Predicted Topic:", predict(headline))
```

---

## 🎯 Results  
- **Evaluation Metric:** Accuracy & F1-score  
- Trained with HuggingFace `transformers` on AG News  
- Achieved high accuracy in classifying unseen news samples  

---

## 💻 Deployment  
Run the Gradio app for an interactive demo:
```bash
python app/gradio_app.py
```
