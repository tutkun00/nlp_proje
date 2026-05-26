import pandas as pd
import torch
import gc
import matplotlib.pyplot as plt
import os
os.makedirs('grafikler_roberta', exist_ok=True)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

# Verisetini getirme ve training, validation, test verisi olarak ayırma

df = pd.read_csv('dataset.csv')
train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["result"])
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42, stratify=train_val_df["result"])

train_text1, train_text2, train_labels = train_df['sentence_1'].tolist(), train_df['sentence_2'].tolist(), train_df['result'].tolist()

val_text1, val_text2, val_labels = val_df['sentence_1'].tolist(), val_df['sentence_2'].tolist(), val_df['result'].tolist()

test_text1, test_text2, test_labels = test_df['sentence_1'].tolist(), test_df['sentence_2'].tolist(), test_df['result'].tolist()


# Tokenize işlemi 

print("Tokenizer islemi basliyor.")
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', num_labels=4)

new_tokens = ["[AST-1]", "[AST-2]", "[AST-3]", "[AST-4]","[AST-5]", "[AST-6]","[AST-7]", "[AST-8]", "[AST-9]", "[AST-10]"]
tokenizer.add_tokens(new_tokens)

train_encodings = tokenizer(train_text1, train_text2, truncation=True, padding='max_length', max_length=128)
val_encodings = tokenizer(val_text1, val_text2, truncation=True, padding='max_length', max_length=128)
test_encodings = tokenizer(test_text1, test_text2, truncation=True, padding='max_length', max_length=128)


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

# %%


# Hiperparametre ayarlaması

learning_rates = [5e-6, 1e-5, 2e-5, 3e-5, 5e-5]
batch_sizes = [4, 8, 12, 16, 24]
epochs_list = [2, 3, 4, 5, 6]

best_bs = 8
best_epoch = 4
best_lr = 2e-5



min_loss = float('inf')
lr_train_losses = []
lr_val_losses = []

for lr in learning_rates:
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=4)
    model.resize_token_embeddings(len(tokenizer))
    
    training_args = TrainingArguments(output_dir="./temp", num_train_epochs=best_epoch, per_device_train_batch_size=best_bs, per_device_eval_batch_size=best_bs, learning_rate=lr, eval_strategy="epoch", logging_strategy="epoch", save_strategy="no", report_to="none")
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset)
    trainer.train()
    
    history = trainer.state.log_history
    t_loss = [h['loss'] for h in history if 'loss' in h]
    v_loss = [h['eval_loss'] for h in history if 'eval_loss' in h]
    
    final_t = t_loss[-1] if len(t_loss) > 0 else 0
    final_v = v_loss[-1] if len(v_loss) > 0 else 0
    lr_train_losses.append(final_t)
    lr_val_losses.append(final_v)
    
    if final_v < min_loss:
        min_loss = final_v
        best_lr = lr
        
    del model, trainer; torch.cuda.empty_cache(); gc.collect()


plt.figure(figsize=(7,5))
plt.plot([str(lr) for lr in learning_rates], lr_train_losses, label='Train Loss', marker='o', color='blue')
plt.plot([str(lr) for lr in learning_rates], lr_val_losses, label='Val Loss', marker='o', color='red')
plt.title('Learning Rate - Overfit Analizi')
plt.xlabel('Learning Rate Değerleri')
plt.ylabel('Kayıp (Loss)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('grafikler_roberta/1_LearningRate_Grafik.png')
plt.close()




min_loss = float('inf')
bs_train_losses = []
bs_val_losses = []

for bs in batch_sizes:
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=4)
    model.resize_token_embeddings(len(tokenizer))
    
    training_args = TrainingArguments(output_dir="./temp", num_train_epochs=best_epoch, per_device_train_batch_size=bs, per_device_eval_batch_size=bs, learning_rate=best_lr, eval_strategy="epoch", logging_strategy="epoch", save_strategy="no", report_to="none")
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset)
    trainer.train()
    
    history = trainer.state.log_history
    t_loss = [h['loss'] for h in history if 'loss' in h]
    v_loss = [h['eval_loss'] for h in history if 'eval_loss' in h]
    
    final_t = t_loss[-1] if len(t_loss) > 0 else 0
    final_v = v_loss[-1] if len(v_loss) > 0 else 0
    bs_train_losses.append(final_t)
    bs_val_losses.append(final_v)
    
    if final_v < min_loss:
        min_loss = final_v
        best_bs = bs
        
    del model, trainer; torch.cuda.empty_cache(); gc.collect()


plt.figure(figsize=(7,5))
plt.plot([str(bs) for bs in batch_sizes], bs_train_losses, label='Train Loss', marker='o', color='blue')
plt.plot([str(bs) for bs in batch_sizes], bs_val_losses, label='Val Loss', marker='o', color='red')
plt.title('Batch Size - Overfit Analizi')
plt.xlabel('Batch Size Değerleri')
plt.ylabel('Kayıp (Loss)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('grafikler_roberta/2_BatchSize_Grafik.png')
plt.close()


min_loss = float('inf')
ep_train_losses = []
ep_val_losses = []

for ep in epochs_list:
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=4)
    model.resize_token_embeddings(len(tokenizer))
    
    training_args = TrainingArguments(output_dir="./temp", num_train_epochs=ep, per_device_train_batch_size=best_bs, per_device_eval_batch_size=best_bs, learning_rate=best_lr, eval_strategy="epoch", logging_strategy="epoch", save_strategy="no", report_to="none")
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset)
    trainer.train()
    
    history = trainer.state.log_history
    t_loss = [h['loss'] for h in history if 'loss' in h]
    v_loss = [h['eval_loss'] for h in history if 'eval_loss' in h]
    
    final_t = t_loss[-1] if len(t_loss) > 0 else 0
    final_v = v_loss[-1] if len(v_loss) > 0 else 0
    ep_train_losses.append(final_t)
    ep_val_losses.append(final_v)
    
    if final_v < min_loss:
        min_loss = final_v
        best_epoch = ep
        
    del model, trainer; torch.cuda.empty_cache(); gc.collect()


plt.figure(figsize=(7,5))
plt.plot([str(ep) for ep in epochs_list], ep_train_losses, label='Train Loss', marker='o', color='blue')
plt.plot([str(ep) for ep in epochs_list], ep_val_losses, label='Val Loss', marker='o', color='red')
plt.title('Epoch - Overfit Analizi')
plt.xlabel('Epoch Değerleri')
plt.ylabel('Kayıp (Loss)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('grafikler_roberta/3_Epoch_Grafik.png')
plt.close()

print(f"best_epoch:{best_epoch}, best_lr:{best_lr}, best_bs:{best_bs}")

# %%


# En iyi hiperparametreler ile eğitme

model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=4)
model.resize_token_embeddings(len(tokenizer))



training_args = TrainingArguments(
    output_dir="./temp", num_train_epochs=best_epoch, per_device_train_batch_size=best_bs, per_device_eval_batch_size=best_bs, 
    learning_rate=best_lr, eval_strategy="no", logging_strategy="epoch", save_strategy="no", report_to="none", warmup_ratio=0.1, weight_decay=0.01
)


vt_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=vt_dataset,
)

trainer.train()

predictions_train = trainer.predict(train_dataset) 
bert_preds_train = predictions_train.predictions.argmax(axis=1)

predictions_test = trainer.predict(test_dataset)
bert_preds_test = predictions_test.predictions.argmax(axis=1)




# Test kısmı
print("-----------------------------------------------------------------------------------------------------------\n")   
print("\n--- 1. MODEL EGITIM BASARISI ---")
print(f"Model Accuracy (Doğruluk): %{accuracy_score(train_labels, bert_preds_train) * 100:.2f}")
print("-----------------------------------------------------------------------------------------------------------\n")   
print("\n--- 1. MODEL TEST BASARISI ---")
print(f"Model Accuracy (Doğruluk): %{accuracy_score(test_labels, bert_preds_test) * 100:.2f}")
print("-----------------------------------------------------------------------------------------------------------\n") 
print("\nDETAYLI SINIFLANDIRMA RAPORU (Precision, Recall, F1-Score):")
print(classification_report(test_labels, bert_preds_test))
print("-----------------------------------------------------------------------------------------------------------\n") 


# Confusion Matrix çizdirme
cm = confusion_matrix(test_labels, bert_preds_test)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['0-Sorun Yok', '1-Tutarsizlik',
                    '2-Bilissel Boz.', '3-Her Ikisi']
).plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title('RoBERTa — Confusion Matrix', fontsize=13)
plt.tight_layout()
plt.savefig('grafikler_roberta/bert_confusion_matrix.png', dpi=150)
plt.close()

            


# Nihai modeli oluşturup kaydetme
all_text1 = df['sentence_1'].tolist()
all_text2 = df['sentence_2'].tolist()
all_labels = df['result'].tolist()

all_encodings = tokenizer(all_text1, all_text2, truncation=True, padding=True, max_length=128)
all_dataset = AstronautDataset(all_encodings, all_labels)

production_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=4)
production_model.resize_token_embeddings(len(tokenizer))

production_args = TrainingArguments(
    output_dir="./temp", num_train_epochs=best_epoch, per_device_train_batch_size=best_bs, per_device_eval_batch_size=best_bs, 
    learning_rate=best_lr, eval_strategy="no", logging_strategy="epoch", save_strategy="no", report_to="none", warmup_ratio=0.1, weight_decay=0.01
)

production_trainer = Trainer(
    model=production_model,
    args=production_args,
    train_dataset=all_dataset,
)

production_trainer.train()


path = 'D:/model/astronaut_roberta_model'
production_trainer.save_model(path)
tokenizer.save_pretrained(path)


