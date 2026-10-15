import os
import re
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

def reduce_repeated_chars(text):
    """
    3 veya daha fazla tekrar eden harfleri teke indirir.
    Örnek: "çoookkkk" -> "çok", "tamammm" -> "tamam"
    """
    return re.sub(r'(.)\1{2,}', r'\1', text)

def clean_text(text):
    """
    Metindeki gereksiz boşlukları, satır atlamalarını ve harf uzatmalarını temizler.
    """
    text = str(text)
    text = text.replace('\n', ' ')
    text = reduce_repeated_chars(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Eğitim veri seti yolu
train_path = r'c:\Users\akcic\Desktop\konteyner\cv_scoring\yeni_finetune\veriseti\train.csv'
df_train = pd.read_csv(train_path)
df_train = df_train.dropna(subset=['cv_text', 'job_text', 'label'])

train_examples = []
for _, row in df_train.iterrows():
    cv_clean = "query: " + clean_text(row['cv_text'])
    job_clean = "passage: " + clean_text(row['job_text'])
    label = float(row['label'])
    
    train_examples.append(InputExample(texts=[cv_clean, job_clean],label=label))

print(f"Toplam {len(train_examples)} adet E5 eğitim paketi hazırlandı.")

train_loader = DataLoader(train_examples,shuffle=True,batch_size=8)

device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"Kullanılan Donanım: {device_name}")
print("Baz model yükleniyor...")
model = SentenceTransformer("intfloat/multilingual-e5-small")
model.max_seq_length = 512

train_loss = losses.CosineSimilarityLoss(model)

num_epochs = 3
total_steps = len(train_loader) * num_epochs
warmup_steps = int(total_steps * 0.1)

output_model_path = r'c:\Users\akcic\Desktop\konteyner\cv_scoring\yeni_finetune\multilingual-e5-small'

print(f"Eğitim başlatılıyor... (Toplam Adım: {total_steps}, Isınma: {warmup_steps}, Token: 512)")

model.fit(
    train_objectives=[(train_loader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path=output_model_path,
    show_progress_bar=True,
    use_amp=True)

print(f"\nE5 Model eğitimi başarıyla tamamlandı! Model kaydedildi: {output_model_path}")
