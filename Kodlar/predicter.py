import torch
import joblib
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.model_selection import train_test_split
 
sentence_1 = "[AST-1] I've been working about 50% with AST-2 today on pre-pack operations."
sentence_2 = "[AST-2] I've been working about 100% alone today; AST-1 was not with me."
 
LABEL_MAP = {
    0: "Sorun Yok",
    1: "Tutarsızlık",
    2: "Bilişsel Bozulma",
    3: "Her İkisi",
}
 
print("=" * 55)
print("  1. NAİVE BAYES")
print("=" * 55)
 

df = pd.read_csv("dataset.csv")
df["full_sentence"] = df["sentence_1"] + " [SEP] " + df["sentence_2"]
X_train, _, y_train, _ = train_test_split(
    df["full_sentence"], df["result"], test_size=0.2, random_state=42, stratify=df["result"]
)
 
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=3, sublinear_tf=True)
tfidf.fit(X_train)
 
nb_model = joblib.load("./astronaut_naivebayes_model")
 
combined = sentence_1 + " [SEP] " + sentence_2
nb_vec   = tfidf.transform([combined])
nb_pred  = nb_model.predict(nb_vec)[0]
nb_probs = nb_model.predict_proba(nb_vec)[0]
 
print(f"  Cümle 1 : {sentence_1}")
print(f"  Cümle 2 : {sentence_2}")
print(f"  Tahmin  : {nb_pred} — {LABEL_MAP[nb_pred]}")
print(f"  Güven   : %{nb_probs[nb_pred]*100:.2f}")
print()
 
# ─── 2. RoBERTa ──────────────────────────────────────────────
print("=" * 55)
print("  2. RoBERTa")
print("=" * 55)
 
tokenizer = RobertaTokenizer.from_pretrained("./astronaut_roberta_model")
model     = RobertaForSequenceClassification.from_pretrained("./astronaut_roberta_model")
model.eval()
 
encoding = tokenizer(
    sentence_1, sentence_2,
    truncation=True, padding="max_length", max_length=128, return_tensors="pt"
)
 
with torch.no_grad():
    logits = model(**encoding).logits
 
probs      = torch.softmax(logits, dim=1).squeeze().tolist()
rb_pred    = int(torch.argmax(logits).item())
 
print(f"  Cümle 1 : {sentence_1}")
print(f"  Cümle 2 : {sentence_2}")
print(f"  Tahmin  : {rb_pred} — {LABEL_MAP[rb_pred]}")
print(f"  Güven   : %{probs[rb_pred]*100:.2f}")
print()
 
# ─── Karşılaştırma ───────────────────────────────────────────
print("=" * 55)
print("  KARŞILAŞTIRMA")
print("=" * 55)
print(f"  Naive Bayes : {nb_pred} — {LABEL_MAP[nb_pred]}")
print(f"  RoBERTa     : {rb_pred} — {LABEL_MAP[rb_pred]}")
match = "✅ Uyuşuyor" if nb_pred == rb_pred else "❌ Ayrılıyor"
print(f"  Sonuç       : {match}")

