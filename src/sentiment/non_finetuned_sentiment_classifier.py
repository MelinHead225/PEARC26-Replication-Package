import pandas as pd
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import logging
import warnings

# Suppress warnings
logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load & Filter Data
df = pd.read_csv(
    "casse_labeled_dataset.csv"
)

allowed_labels = ["non-negative", "negative"]
df = df[df["Sentiment"].isin(allowed_labels)].reset_index(drop=True)

label2id = {label: i for i, label in enumerate(allowed_labels)}
df["label"] = df["Sentiment"].map(label2id)

# 3. Tokenizer & Dataset
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 4. Stratified Split (same as fine-tuned setup)
train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED
)

test_dataset = SentimentDataset(
    test_df["Comment"], test_df["label"], tokenizer
)
test_loader = DataLoader(test_dataset, batch_size=64)

# 5. Load PRETRAINED Model (NO fine-tuning)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(allowed_labels),
    ignore_mismatched_sizes=True
).to(device)

model.eval()

# 6. Test Evaluation (Zero-shot / Non-fine-tuned)
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].tolist()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()

        all_labels.extend(labels)
        all_preds.extend(preds)

test_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
print("\nNon-fine-tuned Test Accuracy:", test_acc)

id2label = {v: k for k, v in label2id.items()}
target_names = [id2label[i] for i in range(len(id2label))]

print("\nClassification Report (Non-fine-tuned):")
print(classification_report(all_labels, all_preds, target_names=target_names))
