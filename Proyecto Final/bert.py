import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
import torch
from transformers import BertForSequenceClassification, AdamW, get_scheduler
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datasets import Dataset
import os

file = "df_total.csv"
df = pd.read_csv(file)

# print("\nDistribución de las clases en 'Type':")
# print(df['Type'].value_counts())

# Codificar las etiquetas como números
label_encoder = LabelEncoder()
df['Type_encoded'] = label_encoder.fit_transform(df['Type'])


# Ver el mapeo de las etiquetas
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

pd.to_pickle(label_mapping, "label_mapping.pkl")
print("Mapeo de etiquetas:")
print(label_mapping)

# Dividir el dataset en entrenamiento y validación
# Dividir el dataset en entrenamiento, validación y prueba
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df['news'], df['Type_encoded'], test_size=0.3, stratify=df['Type_encoded'], random_state=42
)

val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

train_df = pd.DataFrame({'text': train_texts, 'label': train_labels})
max_count = train_df['label'].value_counts().max()

balanced_train_df = train_df.groupby('label').apply(lambda x: x.sample(max_count, replace=True)).reset_index(drop=True)

print("\nDistribución de las clases en el conjunto de entrenamiento:")
print(balanced_train_df['label'].value_counts())

train_texts = balanced_train_df['text']
train_labels = balanced_train_df['label']


print("\nTamaño del conjunto de entrenamiento:", len(train_texts))
print("Tamaño del conjunto de validación:", len(val_texts))
print("Tamaño del conjunto de prueba:", len(test_texts))


# Crear un Dataset de Hugging Face
train_dataset = Dataset.from_pandas(pd.DataFrame({'text': train_texts, 'label': train_labels}))
val_dataset = Dataset.from_pandas(pd.DataFrame({'text': val_texts, 'label': val_labels}))
test_dataset = Dataset.from_pandas(pd.DataFrame({'text': test_texts, 'label': test_labels}))

# Cargar el tokenizer de BERT
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

#Tokenizar
train_dataset = train_dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length'), batched=True)
val_dataset = val_dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length'), batched=True)
test_dataset = test_dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length'), batched=True)

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

#Comenzamos con bert
# Determinar el número de clases
num_classes = len(label_mapping)

# Cargar el modelo BERT preentrenado
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=num_classes  # Número de clases
)

# Mover el modelo a GPU si está disponible
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
torch.save(device, "device.pt")


print(f"Modelo configurado y movido a {device}.")

# Configurar el optimizador
optimizer = AdamW(model.parameters(), lr=5e-5)

# Crear un scheduler para la tasa de aprendizaje
num_training_steps = len(train_dataset) // 16 * 5  # 3 epochs, batch size de 16
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

print("Optimizador y scheduler configurados.")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

print("DataLoaders creados con éxito.")

if not os.path.exists("bert_classificador"):
    



    def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=5):
        loss_fn = CrossEntropyLoss()

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Fase de entrenamiento
            model.train()
            train_loss = 0
            cont = 0
            cont2 = 0
            for batch in tqdm(train_loader):
                print(f"Batch {cont} de {len(train_loader)}")   
                cont += 1         
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            print(f"Loss de entrenamiento: {avg_train_loss:.4f}")

            # Fase de validación
            model.eval()
            val_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in val_loader:
                    print(f"Batch {cont2} de {len(val_loader)}")
                    cont2 += 1
                    # Verificar si batch es una lista y convertirla en un diccionario
                    # if isinstance(batch, list):
                    #     batch = {key: torch.stack([d[key] for d in batch]) for key in batch[0].keys()}
                    
                    # # Mover los datos al dispositivo
                    # batch = {k: v.to(device) for k, v in batch.items()}
                    
                    # outputs = model(**batch)
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["label"].to(device)

                    
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss

                    val_loss += loss.item()
                    preds = torch.argmax(outputs.logits, dim=-1)
                    correct += (preds == batch['label']).sum().item()
                    total += batch['label'].size(0)

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total
            print(f"Loss de validación: {avg_val_loss:.4f}")
            print(f"Exactitud de validación: {val_accuracy:.4f}")

    # Entrenar el modelo
    train_model(model, train_loader, val_loader, optimizer, scheduler)

    model.save_pretrained("bert_classificador")
    tokenizer.save_pretrained("bert_classificador")

    

    print("Modelo y tokenizer guardados en el directorio 'bert_classificador'.")

    

    


else:
    model, tokenizer = BertForSequenceClassification.from_pretrained("bert_classificador"), AutoTokenizer.from_pretrained("bert_classificador")
    



# Evaluar el modelo en el conjunto de prueba
def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            preds = torch.argmax(outputs.logits, dim=-1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch['label'].cpu().numpy())

    # Generar el reporte de clasificación
    print("\nReporte de clasificación:")
    print(classification_report(true_labels, predictions, target_names=label_mapping.keys()))

    # Generar la matriz de confusión
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
    plt.xlabel('Predicciones')
    plt.ylabel('Etiquetas Reales')
    plt.title('Matriz de Confusión')
    plt.show()

# Evaluar en el conjunto de prueba
print("\nEvaluación en el conjunto de prueba:")
evaluate_model(model, test_loader)


