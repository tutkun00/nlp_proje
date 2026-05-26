import pandas as pd
import matplotlib.pyplot as plt
import os
os.makedirs('grafikler_naivebayes', exist_ok=True)
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score


df = pd.read_csv("dataset.csv")


df['full_sentence'] = df['sentence_1'] + ' [SEP] ' + df['sentence_2']


label_dist = df['result'].value_counts().sort_index()

label_dist.plot(kind='bar', color=['green', 'red', 'orange', 'purple'])
plt.title('Veri Setindeki Sınıf Dağılımı')
plt.xlabel('Sınıf (Label)')
plt.ylabel('Satır Sayısı')
plt.xticks(rotation=0)
plt.savefig('grafikler_naivebayes/sinif_dagilimi.png')



X_train, X_test, y_train, y_test = train_test_split(
    df['full_sentence'], df['result'],
    test_size=0.2,      
    random_state=42,    
    stratify=df['result']
)


tfidf = TfidfVectorizer(
    ngram_range=(1, 2),  
    max_features=5000,  
    min_df=3,           
    sublinear_tf=True    
)

X_train_vec = tfidf.fit_transform(X_train)  
X_test_vec  = tfidf.transform(X_test)


model = MultinomialNB(alpha=1)


model.fit(X_train_vec, y_train)


ytrain_pred = model.predict(X_train_vec)
ytest_pred = model.predict(X_test_vec)

print("-----------------------------------------------------------------------------------------------------------\n")   
print("\n--- 1. MODEL EGITIM BASARISI ---")
print(f"Model Accuracy (Doğruluk): %{accuracy_score(y_train, ytrain_pred) * 100:.2f}")
print("-----------------------------------------------------------------------------------------------------------\n")   
print("\n--- 1. MODEL TEST BASARISI ---")
print(f"Model Accuracy (Doğruluk): %{accuracy_score(y_test, ytest_pred) * 100:.2f}")
print("-----------------------------------------------------------------------------------------------------------\n") 
print("\nDETAYLI SINIFLANDIRMA RAPORU (Precision, Recall, F1-Score):")
print(classification_report(y_test, ytest_pred))
print("-----------------------------------------------------------------------------------------------------------\n")



cm = confusion_matrix(y_test, ytest_pred)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm, display_labels=['0\nSorun Yok','1\nTutarsizlik','2\nBilissel\nBoz.','3\nHer Ikisi']).plot(
    ax=ax, cmap='Blues', values_format='d'
)
ax.set_title('Naive Bayes — Confusion Matrix', fontsize=13)
plt.tight_layout()
plt.savefig('grafikler_naivebayes/nb_confusion_matrix.png', dpi=150)
plt.show()

joblib.dump(model, './astronaut_naivebayes_model')