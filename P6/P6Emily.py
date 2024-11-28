import pandas as pd
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
import numpy as np
from imblearn.under_sampling import RandomUnderSampler

# Descargar los recursos de nltk necesarios
# nltk.download('stopwords')
# nltk.download('punkt')

# Función para cargar el léxico de emojis
def load_emoji_lexicon(filepath):
    emoji_df = pd.read_excel(filepath)
    emoji_dict = {}
    for _, row in emoji_df.iterrows():
        emoji_dict[row['emoji']] = row[['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust', 'negative', 'positive']].to_dict()
    return emoji_dict

# Función de preprocesamiento con manejo de negaciones y repeticiones
def preprocess_text_with_emojis_and_negations(text, emoji_dict):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    
    # Manejo de negaciones en español
    negations = {"no": "no", "nunca": "nunca", "nadie": "nadie", "ningún": "ningún",
                 "ninguna": "ninguna", "ninguno": "ninguno", "jamás": "jamás"}
    
    tokens = nltk.word_tokenize(text, language='spanish')
    for i in range(len(tokens) - 1):
        if tokens[i] in negations:
            tokens[i + 1] = "NO_" + tokens[i + 1]
    
    text = ' '.join(tokens)
    
    # Normalización de repeticiones de caracteres
    #text = re.sub(r'(.)\1+', r'\1\1', text)
    
    # Eliminación de caracteres no alfabéticos
    text = re.sub(r'[^a-zA-ZñÑ\s]', '', text)
    tokens = nltk.word_tokenize(text, language='english')
    stopwords = nltk.corpus.stopwords.words('english')
    tokens = [word for word in tokens if word not in stopwords]
    preprocessed_text = ' '.join(tokens)

    for emoji, sentiments in emoji_dict.items():
        if emoji in text:
            for sentiment, value in sentiments.items():
                if value > 0:
                    preprocessed_text += f' {sentiment}'

    return preprocessed_text

# Cargar el archivo xlsx del corpus y del léxico de emojis
print("Cargando el corpus...")
df = pd.read_excel('Rest_Mex_2022.xlsx')
emoji_dict = load_emoji_lexicon('Emojis lexicon.xlsx')

# Asegurarse de que las columnas son de tipo string antes de concatenarlas
df['Title'] = df['Title'].astype(str)
df['Opinion'] = df['Opinion'].astype(str)

print("Preprocesando el texto...")
# Concatenar las columnas de título y opinión
df['text'] = df['Title'] + ' ' + df['Opinion']
df['text'] = df['text'].apply(lambda x: preprocess_text_with_emojis_and_negations(x, emoji_dict))

# Dividir en conjuntos de entrenamiento y prueba
X = df['text']
y = df['Polarity']
X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=True, test_size=0.2, random_state=0)

# Definir los vectorizadores para las representaciones de texto
vectorizer_tfidf = TfidfVectorizer()
vectorizer_bin = CountVectorizer(binary=True)
vectorizer_freq = CountVectorizer()


# Lista de modelos para entrenamiento y evaluación
models = [
    ('Logistic Regression', LogisticRegression(C=5.1,max_iter=100, n_jobs=-1, penalty='l2', solver='saga'), vectorizer_bin),
    ('Logistic Regression', LogisticRegression(C=5.1,max_iter=100, n_jobs=-1, penalty='l2', solver='saga'), vectorizer_freq),
    ('Logistic Regression', LogisticRegression(C=5.1,max_iter=100, n_jobs=-1, penalty='l2', solver='saga'), vectorizer_tfidf),
    ('Naive Bayes', MultinomialNB(), vectorizer_bin),
    ('Naive Bayes', MultinomialNB(), vectorizer_freq),
    ('Naive Bayes', MultinomialNB(), vectorizer_tfidf),
    ('SVM', SVC(kernel='linear', max_iter=200), vectorizer_bin),
    ('SVM', SVC(kernel='linear', max_iter=200), vectorizer_freq),
    ('SVM', SVC(kernel='linear', max_iter=200, random_state=0), vectorizer_tfidf)
]

results = []

print("Entrenando y evaluando los modelos...")
best_model = None
best_f1 = None  

# Entrenamiento y evaluación de cada modelo
for name, model, vectorizer in models:
    # Vectorizar el texto
    results_fold = []
    print(f'Entrenando y evaluando el modelo {name} con {vectorizer.__class__.__name__}')
    for train_index, test_index in KFold(n_splits=5).split(X_train):
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        X_train_vec = vectorizer.fit_transform(X_train_fold)
        X_test_vec = vectorizer.transform(X_test_fold)

        
        # Balancear las clases en el conjunto de entrenamiento
        # under_sampling = RandomUnderSampler(sampling_strategy='majority')
        # X_train_balanced, y_train_balanced = under_sampling.fit_resample(X_train_vec, y_train_fold)
        smote = SMOTE(random_state=0)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_vec, y_train_fold)
        
        # Entrenar el modelo
        model.fit(X_train_balanced, y_train_balanced)
        
        # Predecir las etiquetas para el conjunto de prueba
        y_pred = model.predict(X_test_vec)
        
        # Calcular métricas de evaluación
        f1_macro = f1_score(y_test_fold, y_pred, average='macro')
        
        if best_f1 is None or f1_macro > best_f1:
            best_f1 = f1_macro
            best_model = model


        # Almacenar los resultados
        results_fold.append(f1_macro)
        
        # Imprimir reporte de clasificación y matriz de confusión para cada modelo       
        # print(f'F1 Macro Score: {f1_macro}')
    
    print("Terminado el entrenamiento y evaluación del modelo")

    avg_f1 = np.mean(results_fold)

    print(f'F1 Macro Score: {avg_f1}')

    results.append((name, vectorizer.__class__.__name__, avg_f1))
    

# Mostrar los resultados
results_df = pd.DataFrame(results, columns=['Model', 'Vectorizer', 'F1 Macro Score'])
print(results_df)

# Seleccionar el mejor resultado
best_result = results_df.loc[results_df['F1 Macro Score'].idxmax()]
print(f'Best Model: {best_result["Model"]} with {best_result["Vectorizer"]}, F1 Macro Score: {best_result["F1 Macro Score"]}')

#Probar con el mejor modelo
#Obtener el mejor modelo


X_test_vec = vectorizer_tfidf.transform(X_test)
y_pred = best_model.predict(X_test_vec)

print(f"\nResultados con el conjunto de prueba")
# Imprimir reporte de clasificación y matriz de confusión
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(f'F1 Macro Score: {f1_score(y_test, y_pred, average="macro")}')

