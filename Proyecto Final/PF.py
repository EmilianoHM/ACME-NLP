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
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import os

# Cargar modelo de spaCy para español
nlp = spacy.load("es_core_news_sm")

# Preprocesamiento del texto
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Eliminar URLs
    text = re.sub(r"[^a-záéíóúñ ]", "", text)  # Eliminar símbolos especiales
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

@st.cache_data
def split_data(file):
    df = pd.read_csv(file)
    df = df.dropna(subset=["news", "Type"])

    # Preprocesamiento de noticias
    df['Processed_News'] = df['news'].apply(preprocess)

    # Representación TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Processed_News'])
    y = df['Type']

    # División de datos
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

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
def train_bert_model(texts_train, y_train, texts_val, y_val):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased", num_labels=len(set(y_train))
    )
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)

    # Convertir sparse matrices a listas si es necesario
    if isinstance(texts_train, csr_matrix):
        texts_train = texts_train.toarray()
    if isinstance(texts_val, csr_matrix):
        texts_val = texts_val.toarray()

    texts_train = [str(text) for text in texts_train]
    texts_val = [str(text) for text in texts_val]

    def tokenize_function(examples):
        encodings = tokenizer(examples["text"], padding="max_length", truncation=True)
        encodings["label"] = examples["label"]
        return encodings

    train_dataset = Dataset.from_dict({"text": texts_train, "label": y_train_enc})
    val_dataset = Dataset.from_dict({"text": texts_val, "label": y_val_enc})

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epochs = 1
    cont = 0
    for epoch in range(epochs):
        print("Epoch", epoch + 1)
        model.train()
        for batch in train_dataloader:
            print("Batch", cont + 1, "de", len(train_dataloader))
            cont += 1
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    # Guardar modelo y tokenizer
    model.save_pretrained("bert_model")
    tokenizer.save_pretrained("bert_model")
    
    # Guardar el label encoder
    pd.to_pickle(label_encoder, "label_encoder.pkl")

    return model, tokenizer, label_encoder

# Modificación en la sección de clasificación
def load_bert_model():
    model = AutoModelForSequenceClassification.from_pretrained("bert_model")
    tokenizer = AutoTokenizer.from_pretrained("bert_model")
    label_encoder = pd.read_pickle("label_encoder.pkl")
    return model, tokenizer, label_encoder

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
        st.success(f"Baseline entrenado. Validación Accuracy: {val_acc_baseline:.2f}, Prueba Accuracy: {test_acc_baseline:.2f}")

        # Fine-tuning de BERT
        if not os.path.exists("bert_model"):
            st.write("Entrenando el modelo BERT...")
            bert_model, bert_tokenizer, label_encoder = train_bert_model(
                X_train, y_train, X_val, y_val
            )
            st.success("Modelo BERT entrenado exitosamente.")
        else:
            st.write("Cargando el modelo BERT...")
            bert_model, bert_tokenizer, label_encoder = load_bert_model()
            st.success("Modelo BERT cargado exitosamente.")

        st.subheader("Prueba los Modelos")
        user_input = st.text_area("Escribe una noticia para clasificarla:")
        if st.button("Clasificar"):
            if user_input:
                baseline_pred, bert_pred = classify_news(user_input, vectorizer, baseline_model, bert_model, bert_tokenizer, label_encoder)

                st.write("### Resultado del Baseline:")
                st.success(f"Categoría Predicha (Baseline): **{baseline_pred}**")
                st.write("### Resultado del BERT:")
                st.success(f"Categoría Predicha (BERT): **{bert_pred}**")
                # processed_input = preprocess(user_input)
                # vectorized_input = vectorizer.transform([processed_input])
                # baseline_prediction = baseline_model.predict(vectorized_input)

                # encoded_input = bert_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
                # bert_output = bert_model(**encoded_input)
                # bert_prediction = torch.argmax(bert_output.logits, axis=1).item()
                # bert_category = label_encoder.inverse_transform([bert_prediction])[0]

                # st.write("### Resultado del Baseline:")
                # st.success(f"Categoría Predicha (Baseline): **{baseline_prediction[0]}**")
                # st.write("### Resultado del BERT:")
                # st.success(f"Categoría Predicha (BERT): **{bert_category}**")

if __name__ == "__main__":
    main()
