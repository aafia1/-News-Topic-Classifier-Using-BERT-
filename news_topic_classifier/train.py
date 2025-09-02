import os
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from utils.labels import LABELS

def main():
    model_name = os.environ.get("MODEL_NAME", "bert-base-uncased")
    output_dir = os.environ.get("OUTPUT_DIR", "./models/news_topic_classifier")
    num_epochs = float(os.environ.get("NUM_EPOCHS", 3))
    lr = float(os.environ.get("LR", 2e-5))
    per_device_train_bs = int(os.environ.get("TRAIN_BS", 16))
    per_device_eval_bs = int(os.environ.get("EVAL_BS", 16))
    max_length = int(os.environ.get("MAX_LEN", 64))

    print("Loading dataset...")
    dataset = load_dataset("ag_news")

    # Keep only headline + label
    def select_cols(batch):
        texts = []
        for i in range(len(batch['label'])):
            title = batch['title'][i] if 'title' in batch else None
            text = batch['text'][i]
            headline = title if (title is not None and title != '') else text
            texts.append(headline)
        return {"headline": texts, "label": batch["label"]}

    dataset = dataset.map(select_cols, batched=True, remove_columns=dataset["train"].column_names)

    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["headline"], truncation=True, padding=False, max_length=max_length)

    encoded = dataset.map(tokenize, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("Loading model...")
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(LABELS))

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="weighted")
        return {"accuracy": acc, "f1": f1}

    args = TrainingArguments(
        output_dir="./models",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=per_device_train_bs,
        per_device_eval_batch_size=per_device_eval_bs,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()

    print("Evaluating...")
    metrics = trainer.evaluate()
    print(metrics)

    print(f"Saving to {output_dir} ...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(os.path.join(output_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

if __name__ == "__main__":
    main()
