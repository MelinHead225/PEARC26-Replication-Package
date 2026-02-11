import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import itertools
from transformers import logging
from huggingface_hub import HfApi, HfFolder, upload_folder

# Suppress warnings
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load & Filter Data
df = pd.read_csv("/bsuhome/ericmelin/ORNL/ORNL-Project-2/Multi-Artifact-SATD-Prioritization/src/sentiment/casse_labeled_dataset.csv")

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

# 4. Stratified Split
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED)

train_dataset = SentimentDataset(train_df["Comment"], train_df["label"], tokenizer)
val_dataset   = SentimentDataset(val_df["Comment"], val_df["label"], tokenizer)
test_dataset  = SentimentDataset(test_df["Comment"], test_df["label"], tokenizer)

# 5. Grid Search Hyperparameters
# learning_rates = [1e-5, 2e-5, 3e-5]
learning_rates = [3e-5]
batch_sizes = [64]
# weight_decays = [0.0, 0.01]
weight_decays = [0.01]


epochs = 10
patience = 2  # early stopping

def train_and_eval(lr, batch_size, weight_decay):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Load model with 2 labels, replacing the 3-class head
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(allowed_labels),
        ignore_mismatched_sizes=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_val = 0.0
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total

        if val_acc > best_val:
            best_val = val_acc
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_val

# 6. Run Grid Search
best_acc = 0.0
best_params = None
for lr, batch_size, weight_decay in itertools.product(learning_rates, batch_sizes, weight_decays):
    print(f"Trying lr={lr}, batch={batch_size}, wd={weight_decay}")
    val_acc = train_and_eval(lr, batch_size, weight_decay)
    print(f"Validation Accuracy = {val_acc:.4f}")
    if val_acc > best_acc:
        best_acc = val_acc
        best_params = (lr, batch_size, weight_decay)

print("\nBest Validation Accuracy:", best_acc)
print("Best Params:", best_params)
best_lr, best_batch, best_wd = best_params

# 7. Train Final Model Using Best Params
train_loader = DataLoader(train_dataset, batch_size=best_batch, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=best_batch)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(allowed_labels),
    ignore_mismatched_sizes=True
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=best_lr, weight_decay=best_wd)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

# 8. Test Evaluation
all_preds, all_labels = [], []

model.eval()
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].tolist()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
        all_labels.extend(labels)
        all_preds.extend(preds)

test_acc = sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)
print("\nTest Accuracy:", test_acc)

id2label = {v: k for k, v in label2id.items()}
target_names = [id2label[i] for i in range(len(id2label))]
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=target_names))

# 9. Save Final Model
save_dir = "twitter-roberta-base-sentiment-latest"

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# Upload to Hugging Face Hub
repo_name = "fine-tuned-sentiment-roberta"   # Choose any name for your model

api = HfApi()

username = api.whoami()["name"]
full_repo_id = f"{username}/{repo_name}"

# Create the repo (skip if it already exists)
api.create_repo(full_repo_id, exist_ok=True)

# Upload entire model folder
upload_folder(
    folder_path=save_dir,
    repo_id=full_repo_id,
    commit_message="Upload fine-tuned sentiment BERT model"
)

print(f"\nModel uploaded successfully to: https://huggingface.co/{full_repo_id}")