import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE
from gensim.models import Word2Vec
from collections import Counter
import numpy as np

# Descargar recursos necesarios para procesamiento de texto
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('punkt_tab')

# Cargar el corpus
print("Cargando el corpus...")
corpus = pd.read_excel('Rest_Mex_2022.xlsx')  # Cambia por tu archivo
print(f"Corpus cargado con {corpus.shape[0]} filas y {corpus.shape[1]} columnas.")

# Concatenar Title y Opinion
print("Concatenando las columnas Title y Opinion...")
corpus['Title'] = corpus['Title'].astype(str)
corpus['Opinion'] = corpus['Opinion'].astype(str)
corpus['Text'] = corpus['Title'].fillna('') + ' ' + corpus['Opinion'].fillna('')
corpus = corpus[['Text', 'Polarity']]

# Normalización del texto
print("Normalizando el texto...")

# Configurar stopwords y stemmer para español
spanish_stopwords = set(stopwords.words('spanish'))
stemmer = SnowballStemmer('spanish')

def normalize_text(text):
    text = text.lower()  # Minúsculas
    text = re.sub(r'[^\w\s]', '', text)  # Eliminar puntuación
    text = re.sub(r'\d+', '', text)  # Eliminar números
    words = word_tokenize(text)  # Tokenización
    words = [stemmer.stem(word) for word in words if word not in spanish_stopwords]  # Stemming y stopwords
    return ' '.join(words)

corpus['Text'] = corpus['Text'].apply(normalize_text)
print("Texto normalizado. Ejemplo:")
print(corpus['Text'].head())

# Manejo de negaciones
print("Aplicando manejo de negaciones...")
def handle_negations(text):
    text = re.sub(r'\bno\b\s+(\w+)', r'no_\1', text)  # Unir "no" con la palabra siguiente
    return text

corpus['Text'] = corpus['Text'].apply(handle_negations)

# División de datos
print("Dividiendo los datos en entrenamiento y prueba...")
X_train, X_test, y_train, y_test = train_test_split(
    corpus['Text'], corpus['Polarity'], test_size=0.2, random_state=0, shuffle=True
)
print(f"Conjunto de entrenamiento: {len(X_train)} opiniones.")
print(f"Conjunto de prueba: {len(X_test)} opiniones.")

# Vectorización: Binarized, Frequency, TF-IDF, Embeddings
print("Generando representaciones de texto...")

# Binarized
binarizer = CountVectorizer(binary=True)
X_train_binarized = binarizer.fit_transform(X_train)
X_test_binarized = binarizer.transform(X_test)

# Frequency
frequency_vectorizer = CountVectorizer()
X_train_frequency = frequency_vectorizer.fit_transform(X_train)
X_test_frequency = frequency_vectorizer.transform(X_test)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Word Embeddings
#Dijo el profe que a ese no le hicieramos caso, (podiamos o no podiamos ponerlo)
# print("Entrenando modelo de Word2Vec para embeddings...")
# tokenized_train = [text.split() for text in X_train]
# word2vec_model = Word2Vec(sentences=tokenized_train, vector_size=100, window=5, min_count=1, workers=4)
# def text_to_embeddings(text, model, vector_size):
#     tokens = text.split()
#     embeddings = [model.wv[token] for token in tokens if token in model.wv]
#     if embeddings:
#         return np.mean(embeddings, axis=0)
#     else:
#         return np.zeros(vector_size)

# X_train_embeddings = np.array([text_to_embeddings(text, word2vec_model, 100) for text in X_train])
# X_test_embeddings = np.array([text_to_embeddings(text, word2vec_model, 100) for text in X_test])

# Manejo de desbalanceo
x_tests = [X_test_binarized, X_test_frequency, X_test_tfidf]


print("Aplicando SMOTE para balancear los datos de entrenamiento...")
smote = SMOTE(random_state=0, sampling_strategy='minority', k_neighbors=5)
X_train_binarized_balanced, y_train_binarized_balanced = smote.fit_resample(X_train_binarized, y_train)
X_train_frequency_balanced, y_train_frequency_balanced = smote.fit_resample(X_train_frequency, y_train)
X_train_tfidf_balanced, y_train_tfidf_balanced = smote.fit_resample(X_train_tfidf, y_train)
# print(f"Distribución de clases después de SMOTE: {Counter(y_train_balanced)}")

X_trains = [X_train_binarized_balanced, X_train_frequency_balanced, X_train_tfidf_balanced]
y_trains = [y_train_binarized_balanced, y_train_frequency_balanced, y_train_tfidf_balanced]

# Modelos a evaluar
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=0, multi_class='multinomial'),
    "MLP": MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=100, random_state=0, early_stopping=True)
}

# Entrenamiento y validación
print("Entrenando modelos...")
best_model = None
best_f1 = 0
representations = ['Binarized', 'Frequency', 'TF-IDF']

best_representation = None
x = None
y = None

for model_name, model in models.items():
    print(f"Entrenando {model_name}...")
    for index, x_train in enumerate(X_trains):             
        y_train = y_trains[index]
        print(f"Entrenando {model_name} con {representations[index]}...")
        scores = cross_validate(model, x_train, y_train, cv=5, scoring='f1_macro', return_train_score=False)
        avg_f1 = scores['test_score'].mean()
        print(f"F1 Macro promedio para {model_name}: {avg_f1:.4f}")
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_model = model
            best_representation = representations[index]
            x = x_train
            y = y_train
            x_test = x_tests[index]

# Entrenar el mejor modelo
print(f"El mejor modelo es {best_model.__class__.__name__} con F1 Macro promedio de {best_f1:.4f} usando {best_representation}.")
print("Entrenando el mejor modelo...")
best_model.fit(x, y)

# Evaluación en el conjunto de prueba
print("Evaluando el modelo en el conjunto de prueba...")
y_pred = best_model.predict(x_test)
f1_test = f1_score(y_test, y_pred, average='macro')
print(f"F1 Macro en el conjunto de prueba: {f1_test:.4f}")
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))
