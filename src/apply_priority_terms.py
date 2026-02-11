import json
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import numpy as np

# CONFIG

# INPUT_FILE = "master_artifacts_with_sentiment.jsonl"
INPUT_FILE =  "code_comments.jsonl"
# OUTPUT_FILE = "/master_artifacts_with_sentiment_and_priority_scores.jsonl"
OUTPUT_FILE = "code_comments_with_sentiment.jsonl"

PRIORITY_TERMS = [
    "serious", "junk", "urgent", "not work", "fundamental", "need the correctness",
    "terrible", "wrong", "fugly", "implement properly", "horrible", "improve this a lot",
    "add it now", "fix", "system malfunctioning", "killing", "fail", "slow", "fatal",
    "inefficient", "problematic", "optimize", "issue", "do this more efficiently",
    "bug", "check this result", "error", "nasty", "broken", "might hang", "exception",
    "fragile", "test fail", "ugliest", "fix testcase", "hack", "untested", "fugly code",
    "needs tested", "code violates", "more testes", "silly", "improve this test",
    "don‘t like", "yuck", "does not make sense", "ugly", "this needs work",
    "stupid", "find a cleaner way to do this"
]

# ARTIFACT-AWARE TEXT EXTRACTION

def extract_text(entry):
    atype = entry.get("artifact_type")

    if atype == "comment":
        return entry.get("comment")

    if atype == "Commit":
        src = entry.get("source_sections", {})
        return src.get("message") or src.get("raw_message")

    if atype == "PRSection":
        return entry.get("source_sections", {}).get("text")

    if atype == "IssueSection":
        return entry.get("source_sections", {}).get("text")

    return None


# INITIALIZE TF-IDF

priority_doc = " ".join(PRIORITY_TERMS)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    lowercase=True,
    stop_words="english"
)
priority_vec = vectorizer.fit_transform([priority_doc])


# INITIALIZE EMBEDDINGS

model = SentenceTransformer("all-mpnet-base-v2")
priority_embeddings = model.encode(PRIORITY_TERMS, convert_to_tensor=True)
priority_centroid = priority_embeddings.mean(dim=0)


# SCORING FUNCTION

def compute_priority_scores(text):
    if not text or not str(text).strip():
        return 0.0, 0.0, 0.0

    text_l = text.lower()

    # Lexical score
    lexical_hits = sum(1 for term in PRIORITY_TERMS if term in text_l)
    lexical_score = lexical_hits / len(PRIORITY_TERMS)

    # TF-IDF similarity
    doc_vec = vectorizer.transform([text])
    tfidf_score = float(cosine_similarity(doc_vec, priority_vec)[0][0])

    # Embedding similarity
    doc_emb = model.encode(text, convert_to_tensor=True)
    embedding_score = float(util.cos_sim(doc_emb, priority_centroid).item())

    return lexical_score, tfidf_score, embedding_score

# FIRST PASS: COLLECT SCORES

entries = []

print(f"\nComputing priority features for: {INPUT_FILE}\n")

with open(INPUT_FILE, "r") as f:
    for line in tqdm(f, desc="Priority feature extraction"):
        entry = json.loads(line)

        text = extract_text(entry)

        L, T, E = compute_priority_scores(text)

        entry["priority_features"] = {
            "lexical": float(L),
            "tfidf": float(T),
            "embedding": float(E)
        }

        entries.append(entry)


# WRITE OUTPUT

with open(OUTPUT_FILE, "w") as f:
    for entry in entries:
        f.write(json.dumps(entry) + "\n")

print(f"\nDone! Priority features saved to: {OUTPUT_FILE}\n")
