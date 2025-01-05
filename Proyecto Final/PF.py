# Importación de librerías
import pandas as pd
import re
import spacy
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from transformers import pipeline

# Cargar modelo de spaCy para español
nlp = spacy.load("es_core_news_sm")

# Preprocesamiento del texto
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Eliminar URLs
    text = re.sub(r"[^a-záéíóúñ ]", "", text)  # Eliminar símbolos especiales
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

# Entrenamiento del baseline (Regresión Logística con TF-IDF)
@st.cache_data
def train_baseline(file):
    df = pd.read_csv(file)
    df = df.dropna(subset=["news", "Type"])

    # Preprocesamiento de noticias
    df['Processed_News'] = df['news'].apply(preprocess)

    # Representación TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Processed_News'])
    y = df['Type']

    # División del conjunto de datos
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # 80% entrenamiento, 20% restante
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp  # 10% validación, 10% pruebas
    )

    

    # Entrenamiento de Regresión Logística
    model = LogisticRegression(C=5.1, max_iter=100, n_jobs=-1, penalty='l2', solver='saga')
    model.fit(X_train, y_train)

    # Evaluación en validación y pruebas
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    report = classification_report(y_test, y_test_pred, output_dict=True)

    # Mostrar resultados
    print(f"Accuracy en Validación: {val_acc:.2f}")
    print(f"Accuracy en Pruebas: {test_acc:.2f}")
    
    return model, vectorizer, val_acc, test_acc, report

# Clasificación usando LLM (Hugging Face Transformers)
@st.cache_data
def get_llm_classifier():
    classifier = pipeline("text-classification", model="bert-base-multilingual-cased", tokenizer="bert-base-multilingual-cased")
    return classifier

# Interfaz con Streamlit
def main():
    st.title("Clasificación de Noticias - Baseline vs LLM")
    st.write("Este proyecto compara un modelo de *Regresión Logística* (baseline) con un *LLM* para clasificar noticias.")

    # Subir archivo CSV
    uploaded_file = st.file_uploader("Sube tu archivo CSV con columnas 'News' y 'Type'", type=["csv"])
    if uploaded_file:
        st.write("Archivo cargado exitosamente. Entrenando el modelo baseline...")

        # Entrenamiento del baseline
        baseline_model, vectorizer, baseline_val_acc, baseline_test_acc, baseline_report = train_baseline(uploaded_file)
        st.success(f"Baseline entrenado con un Accuracy en Validación del {baseline_val_acc:.2f} y en Pruebas del {baseline_test_acc:.2f}")
        st.subheader("Reporte de Clasificación - Baseline:")
        st.write(pd.DataFrame(baseline_report).T)

        # Inicializar LLM
        st.write("Cargando el modelo LLM (BERT)...")
        llm_classifier = get_llm_classifier()
        st.success("Modelo LLM cargado exitosamente.")

        # Clasificación con ambos modelos
        st.subheader("Prueba los Modelos")
        user_input = st.text_area("Escribe una noticia para clasificarla:")
        if st.button("Clasificar"):
            if user_input:
                # Baseline Prediction
                processed_input = preprocess(user_input)
                vectorized_input = vectorizer.transform([processed_input])
                baseline_prediction = baseline_model.predict(vectorized_input)
                
                # LLM Prediction
                llm_prediction = llm_classifier(user_input)
                
                # Mostrar resultados
                st.write("### Resultado del Baseline:")
                st.success(f"Categoría Predicha (Baseline - Regresión Logística): *{baseline_prediction[0]}*")
                st.write("### Resultado del LLM:")
                st.success(f"Categoría Predicha (LLM - BERT): *{llm_prediction[0]['label']}*")
            else:
                st.warning("Por favor, ingresa una noticia.")

if _name_ == "_main_":
    main()
