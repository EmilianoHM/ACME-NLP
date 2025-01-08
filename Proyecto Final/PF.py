import pandas as pd
import re
import spacy
from scipy.sparse import csr_matrix
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import os
from transformers import BertForSequenceClassification, AdamW, get_scheduler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Cargar modelo de spaCy para español
nlp = spacy.load("es_core_news_sm")

#Un df vacío
df = pd.DataFrame()
df2 = pd.DataFrame()

# Preprocesamiento del texto
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Eliminar URLs
    text = re.sub(r"[^a-záéíóúñ ]", "", text)  # Eliminar símbolos especiales
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

@st.cache_data
def split_data(file):
    global df, df2
    df = pd.read_csv(file)
    df2 = df
    df = df.dropna(subset=["news", "Type"])
    # Mostrar distribución de categorías
    st.write("### Distribución de Categorías de Noticias:")
    category_dist = df['Type'].value_counts()
    st.write(pd.DataFrame({
        'Categoría': category_dist.index,
        'Número de Noticias': category_dist.values,
        'Porcentaje': (category_dist.values / len(df) * 100).round(2)
    }))
    # Preprocesamiento de noticias
    df['Processed_News'] = df['news'].apply(preprocess)

    # Representación TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Processed_News'])
    y = df['Type']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=42)
    test_size_adjusted = 0.4  # 10% del total (40% del 25% restante)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size_adjusted, random_state=42)

    # Mostrar distribución del split
    st.write("\n### Distribución del Split de Datos:")
    split_dist = pd.DataFrame({
        'Conjunto': ['Entrenamiento', 'Validación', 'Pruebas'],
        'Número de Muestras': [X_train.shape[0], X_val.shape[0], X_test.shape[0]],
        'Porcentaje': [75, 15, 10]
    })
    st.write(split_dist)

    # Mostrar distribución de categorías en cada conjunto
    st.write("\n### Distribución de Categorías por Conjunto:")
    
    train_dist = pd.Series(y_train).value_counts()
    val_dist = pd.Series(y_val).value_counts()
    test_dist = pd.Series(y_test).value_counts()
    
    dist_df = pd.DataFrame({
        'Entrenamiento': train_dist,
        'Validación': val_dist,
        'Pruebas': test_dist
    }).fillna(0)
    
    st.write(dist_df)

    return df, X_train, X_val, X_test, y_train, y_val, y_test, vectorizer

def train_baseline(X_train, y_train, X_val, y_val, X_test, y_test):
    model = LogisticRegression(C=5.1, max_iter=100, n_jobs=-1, penalty='l2', solver='saga')
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_report = classification_report(y_val, y_val_pred, output_dict=True)

    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_report = classification_report(y_test, y_test_pred, output_dict=True)

    return model, val_acc, val_report, test_acc, test_report

# Entrenamiento manual de BERT
@st.cache_data
def train_bert_model():
    global df2    

    label_encoder = LabelEncoder()
    df2['Type_encoded'] = label_encoder.fit_transform(df2['Type'])


    # Ver el mapeo de las etiquetas
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    pd.to_pickle(label_mapping, "label_mapping.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")
    print("Mapeo de etiquetas:")
    print(label_mapping)

    # Dividir el dataset en entrenamiento y validación
    # Dividir el dataset en entrenamiento, validación y prueba
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df2['news'], df2['Type_encoded'], test_size=0.25, stratify=df2['Type_encoded'], random_state=42
    )

    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.4, stratify=temp_labels, random_state=42
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

    print(f"Modelo configurado y movido a {device}.")

    # Configurar el optimizador
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Crear un scheduler para la tasa de aprendizaje
    num_training_steps = len(train_dataset) // 16 * 3 # 5 epochs, batch size de 16
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    print("Optimizador y scheduler configurados.")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    print("DataLoaders creados con éxito.")

    def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=3):
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
    #Guardar device
    torch.save(device, "device.pt")
    #Guardar test_loader
    torch.save(test_loader, "test_loader.pt")

    

    print("Modelo y tokenizer guardados en el directorio 'bert_classificador'.")    

    return model, tokenizer, label_encoder, label_mapping, test_loader, device


# Modificación en la sección de clasificación
def load_bert_model():
    model, tokenizer = BertForSequenceClassification.from_pretrained("bert_classificador"), AutoTokenizer.from_pretrained("bert_classificador")
    label_mapping = pd.read_pickle("label_mapping.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    device = torch.load("device.pt")
    test_loader = torch.load("test_loader.pt")
    return model, tokenizer, label_encoder, device, test_loader, label_mapping

def evaluate_model(model, test_loader, label_mapping, device):    
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


    return cm, classification_report(true_labels, predictions, target_names=label_mapping.keys(), output_dict=True)
    


def classify_news(user_input, vectorizer, baseline_model, bert_model, bert_tokenizer, label_encoder):
    processed_input = preprocess(user_input)
    vectorized_input = vectorizer.transform([processed_input])
    baseline_prediction = baseline_model.predict(vectorized_input)

    encoded_input = bert_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
    bert_output = bert_model(**encoded_input)
    bert_prediction = torch.argmax(bert_output.logits, axis=1).item()
    bert_category = label_encoder.inverse_transform([bert_prediction])[0]

    return baseline_prediction[0], bert_category


# Interfaz con Streamlit

# Inicializar los estados de sesión
if 'bert_evaluated' not in st.session_state:
    st.session_state['bert_evaluated'] = False
if 'baseline_results' not in st.session_state:
    st.session_state['baseline_results'] = None
if 'bert_matriz' not in st.session_state:
    st.session_state['bert_matriz'] = None
if 'bert_reporte' not in st.session_state:
    st.session_state['bert_reporte'] = None

def main():
    st.title("Clasificación de Noticias - Baseline vs BERT Fine-Tuning")
    uploaded_file = st.file_uploader("Sube tu archivo CSV con columnas 'news' y 'Type'", type=["csv"])
    if uploaded_file:
        st.write("Procesando datos...")

        df, X_train, X_val, X_test, y_train, y_val, y_test, vectorizer = split_data(uploaded_file)

        # Entrenamiento del baseline
        st.write("Entrenando el modelo baseline...")
        baseline_model, val_acc_baseline, val_report_baseline, test_acc_baseline, test_report_baseline = train_baseline(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        st.success("Modelo Baseline entrenado exitosamente.")

        # Fine-tuning de BERT
        if not os.path.exists("bert_classificador"):
            st.write("Entrenando el modelo BERT...")
            bert_model, bert_tokenizer, label_encoder, label_mapping, test_loader, device = train_bert_model()
            st.success("Modelo BERT entrenado exitosamente.")
        else:
            st.write("Cargando el modelo BERT...")
            bert_model, bert_tokenizer, label_encoder, device, test_loader, label_mapping = load_bert_model()
            st.success("Modelo BERT cargado exitosamente.")

        if not st.session_state['bert_evaluated']:
            # Evaluación del modelo BERT
            st.write("Probando modelos en conjunto de prueba...")
            st.write("### Resultados en Baseline:")
            st.write("Matriz de confusión:")
            st.write(confusion_matrix(y_test, baseline_model.predict(X_test)))
            st.write("Reporte de clasificación:")
            st.write(pd.DataFrame(test_report_baseline).T)

            matriz, reporte = evaluate_model(bert_model, test_loader, label_mapping, device)
            st.session_state['bert_matriz'] = matriz
            st.session_state['bert_reporte'] = reporte
            st.session_state['bert_evaluated'] = True

        # Mostrar resultados almacenados
        st.write("### Resultados en BERT:")
        st.write("Matriz de confusión:")
        st.write(st.session_state['bert_matriz'])
        st.write("Reporte de clasificación:")
        st.write(pd.DataFrame(st.session_state['bert_reporte']).T)

        st.subheader("Prueba los Modelos")
        user_input = st.text_area("Escribe una noticia para clasificarla:")
        if st.button("Clasificar"):
            if user_input:
                baseline_pred, bert_pred = classify_news(user_input, vectorizer, baseline_model, bert_model, bert_tokenizer, label_encoder)

                st.write("### Resultado del Baseline:")
                st.success(f"Categoría Predicha (Baseline): *{baseline_pred}*")
                st.write("### Resultado del BERT:")
                st.success(f"Categoría Predicha (BERT): *{bert_pred}*")

if __name__ == "__main__":
    main()
