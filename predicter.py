import torch
from transformers import BertTokenizer, BertForSequenceClassification

klasor_yolu = './astronaout_bert_model' 

tokenizer = BertTokenizer.from_pretrained(klasor_yolu)
model = BertForSequenceClassification.from_pretrained(klasor_yolu)



def durumu_analiz_et(id_1, id_2, log_1, log_2):
    inputs = tokenizer(log_1, log_2, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        nlp_karari = torch.argmax(logits, dim=1).item()
        
    sistem_karari = nlp_karari
    if id_1 == id_2:
        if nlp_karari == 1:
            sistem_karari = 3 
            
    return nlp_karari, sistem_karari


# Senaryo 1: Farklı astronotlar, normal rapor (Beklenen Sistem: 0)
nlp, sistem = durumu_analiz_et(
    id_1="AST-1", id_2="AST-2",
    log_1="The engine made a loud, terrifying bang during the burn.",
    log_2="Checking sensors... it was just a small debris impact on the outer shielding. We are safe."
)
print(f"Senaryo 1 -> NLP: {nlp} | SİSTEM: {sistem}")

# Senaryo 2: Aynı astronot kendiyle çelişiyor (Beklenen Sistem: 3)
nlp, sistem = durumu_analiz_et(
    id_1="AST-3", id_2="AST-3",
    log_1="I am having a very nice conversation with the captain.",
    log_2="The captain died three years ago before we even launched."
)
print(f"Senaryo 2 -> NLP: {nlp} | SİSTEM: {sistem}")

# Senaryo 3: Tereddütlü/Mental bozukluk (Beklenen Sistem: 2 veya 3)
nlp, sistem = durumu_analiz_et(
    id_1="AST-5", id_2="AST-5",
    log_1="Cabin air pressure is stable at 14 PSI.",
    log_2="Cabin air pressure is stable  not at 14 PSI."
)
print(f"Senaryo 3 -> NLP: {nlp} | SİSTEM: {sistem}")

