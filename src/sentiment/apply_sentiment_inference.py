import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Config
HF_MODEL_ID = "MelinHead225/fine-tuned-sentiment-roberta"
INPUT_JSONL = "master_artifacts.jsonl"
OUTPUT_JSONL = "master_artifacts_with_sentiment.jsonl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128

# =========================
# Load model
# =========================
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID).to(DEVICE)
model.eval()

model.config.id2label = {0: "non-negative", 1: "negative"}
model.config.label2id = {"non-negative": 0, "negative": 1}

# Text extraction (strict)
def extract_text_by_artifact(row):
    artifact_type = row.get("artifact_type")

    if artifact_type == "comment":
        val = row.get("comment")
        return val.strip() if val and val.strip() else None

    if artifact_type == "Commit":
        val = row.get("source_sections", {}).get("message")
        return val.strip() if val and val.strip() else None

    if artifact_type in {"PRSection", "IssueSection"}:
        val = row.get("source_sections", {}).get("text")
        return val.strip() if val and val.strip() else None

    return None

# Sentiment prediction
def predict_sentiment(text):
    if not text:
        return None, None

    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()

    return (
        model.config.id2label[pred_id],
        probs[0, pred_id].item()
    )

# Run inference
with open(INPUT_JSONL, "r", encoding="utf-8") as fin, \
     open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:

    for line in fin:
        row = json.loads(line)

        text = extract_text_by_artifact(row)
        label, confidence = predict_sentiment(text)

        row["predicted_sentiment"] = label
        row["predicted_sentiment_confidence"] = confidence

        fout.write(json.dumps(row) + "\n")

print(f"Saved sentiment predictions to {OUTPUT_JSONL}")
