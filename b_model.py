import pandas as pd
import torch
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Verisetini getirme ve training, validation, test verisi olarak ayırma

df = pd.read_csv('temiz_astronaut_dataset.csv')
train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)

train_text1, train_text2, train_labels = train_df['sentence_1'].tolist(), train_df['sentence_2'].tolist(), train_df['result'].tolist()

val_text1, val_text2, val_labels = val_df['sentence_1'].tolist(), val_df['sentence_2'].tolist(), val_df['result'].tolist()

test_text1, test_text2, test_labels = test_df['sentence_1'].tolist(), test_df['sentence_2'].tolist(), test_df['result'].tolist()
test_total_labels = test_df['total_result'].tolist()

# Tokenize işlemi 

print("Tokenizer islemi basliyor.")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(train_text1, train_text2, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_text1, val_text2, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_text1, test_text2, truncation=True, padding=True, max_length=128)


# Veriseti için bir Class

class AstronautDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = AstronautDataset(train_encodings, train_labels)
val_dataset = AstronautDataset(val_encodings, val_labels)
test_dataset = AstronautDataset(test_encodings, test_labels)


# Hiperparametre ayarlaması

learning_rates = [2e-5, 3e-5, 5e-5]  
batch_sizes = [8, 16]                
epochs_list = [3, 4]                

best_accuracy = 0.0
best_params = {}

print("Hiperparametre denemesi baslatiliyor.")

for lr in learning_rates:
    for bs in batch_sizes:
        for epochs in epochs_list:
            print("Model yukleniyor.")
            print(f"\n[DENENIYOR] Learning Rate: {lr} | Batch Size: {bs} | Epochs: {epochs}")
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
            
            training_args = TrainingArguments(
                num_train_epochs=epochs,
                per_device_train_batch_size=bs,
                per_device_eval_batch_size=bs,
                learning_rate=lr,
                eval_strategy="no",     
                save_strategy="no",     
                logging_steps=50,
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
            )
            
            trainer.train()
            predictions = trainer.predict(val_dataset)
            preds = predictions.predictions.argmax(axis=1)
            acc = accuracy_score(val_labels, preds)
            
            print(f"[SONUC] Bu kombinasyonun NLP Basarisi: %{acc * 100:.2f}")
            
            if acc > best_accuracy:
                best_accuracy = acc
                best_params = {'learning_rate': lr, 'batch_size': bs, 'epochs': epochs}
            print("-----------------------------------------------------------------------------------------------------------\n")    
        
            del model
            del trainer
            torch.cuda.empty_cache()
            gc.collect()
            
print(f"En Yuksek NLP Basarisi: %{best_accuracy * 100:.2f}")
print(f"En Iyi Parametreler: {best_params}")
print(f"{best_params} parametreleri ile nihai model egitiliyor.")


# En iyi hiperparametreler ile eğitme

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

training_args = TrainingArguments(
    num_train_epochs=best_params['epochs'],
    per_device_train_batch_size=best_params['batch_size'],
    per_device_eval_batch_size=best_params['batch_size'],
    learning_rate=best_params['learning_rate'],
    eval_strategy="no",     
    save_strategy="no",     
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

predictions_train = trainer.predict(train_dataset) 
bert_preds_train = predictions_train.predictions.argmax(axis=1)

predictions_test = trainer.predict(test_dataset)
bert_preds_test = predictions_test.predictions.argmax(axis=1)



final_system_preds = []

# Astronotlar aynı kişi mi kontrolü

for i in range(len(bert_preds_test)):
    pred = bert_preds_test[i]
    id_1 = test_df.iloc[i]['id_1']
    id_2 = test_df.iloc[i]['id_2']

    if id_1 == id_2:
        if pred == 1:
            final_pred = 3
        else:
            final_pred = pred
    else:
        final_pred = pred

    final_system_preds.append(final_pred)


# Test kısmı

print("-----------------------------------------------------------------------------------------------------------\n")   
print("\n--- 1. MODEL EGITIM BASARISI ---")
print(f"Model Accuracy (Doğruluk): %{accuracy_score(train_labels, bert_preds_train) * 100:.2f}")
print("-----------------------------------------------------------------------------------------------------------\n")   
print("\n--- 1. MODEL TEST BASARISI ---")
print(f"Model Accuracy (Doğruluk): %{accuracy_score(test_labels, bert_preds_test) * 100:.2f}")
print("-----------------------------------------------------------------------------------------------------------\n") 
print("\n--- 2. SISTEM BASARISI ---")
print(f"Sistem Accuracy (Doğruluk): %{accuracy_score(test_total_labels, final_system_preds) * 100:.2f}")
print("-----------------------------------------------------------------------------------------------------------\n") 
print("\nSISTEM ICIN DETAYLI SINIFLANDIRMA RAPORU (Precision, Recall, F1-Score):")
print(classification_report(test_total_labels, final_system_preds))
print("-----------------------------------------------------------------------------------------------------------\n") 
print("\nSISTEM ICIN CONFUSION MATRIX (Karmaşıklık Matrisi):")
print(confusion_matrix(test_total_labels, final_system_preds))
print("-----------------------------------------------------------------------------------------------------------\n") 
            


# Nihai modeli oluşturup kaydetme

all_text1 = df['sentence_1'].tolist()
all_text2 = df['sentence_2'].tolist()
all_labels = df['result'].tolist()

all_encodings = tokenizer(all_text1, all_text2, truncation=True, padding=True, max_length=128)
all_dataset = AstronautDataset(all_encodings, all_labels)

production_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

production_args = TrainingArguments(
    num_train_epochs=best_params['epochs'],
    per_device_train_batch_size=best_params['batch_size'],
    learning_rate=best_params['learning_rate'],           
    eval_strategy="no",     
    save_strategy="no",     
    logging_steps=50,
)

production_trainer = Trainer(
    model=production_model,
    args=production_args,
    train_dataset=all_dataset,
)

production_trainer.train()


path = './astronaout_bert_model'
production_trainer.save_model(path)
tokenizer.save_pretrained(path)


