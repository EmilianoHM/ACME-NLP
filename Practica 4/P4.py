import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import os
from collections import Counter
from itertools import islice
import csv
import random
from collections import defaultdict
from tkinter import messagebox

ngrams_global = []

def roulette_selection(ngrams, ngram_type=2):
        # Obtener las probabilidades
        indice = 4 if ngram_type == 2 else 5
        probabilities = [float(ngram[indice]) for ngram in ngrams]
        
        # Normalizar las probabilidades
        total_prob = sum(probabilities)
        probabilities = [p / total_prob for p in probabilities]

        # Generar un número aleatorio y seleccionar un n-grama basado en las probabilidades
        r = random.random()
        
        cumulative_prob = 0.0
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if r <= cumulative_prob:
                return ngrams[i]
            
def generate_sentence(ngram, data):
        sentence = []

        data = [ngram for ngram in data if '<s>' not in ngram or '</s>' not in ngram]                    

        # Seleccionar el primer n-grama aleatoriamente
        data_inicial = [ngram for ngram in data if '<s>' in ngram]
        data_siguiente = [ngram for ngram in data if '<s>' not in ngram]
        selected_ngram = roulette_selection(data_inicial, ngram)
        if selected_ngram is None:
            return "No se pudo generar un enunciado."

        # Agregar los términos iniciales según el tipo de n-grama
        if ngram == 3:

            sentence.extend([selected_ngram[1], selected_ngram[2]])  # Agrega 2 términos
            context = selected_ngram[2]       
        elif ngram:
            sentence.append(selected_ngram[1])  # Solo agrega 1 término
            context = selected_ngram[1]
        
        # Continuar agregando palabras hasta llegar a </s>
        while True:
           # Seleccionar el siguiente n-grama aleatoriamente
            print("Contexto:",context)
            # for ngram in data_siguiente:
            #     if ngram[0] == context:
            #         print(ngram)
            data_siguiente1 = [ngram for ngram in data_siguiente if ngram[0] == context] 
            print("Data siguiente:",data_siguiente1)           
            selected_ngram = roulette_selection(data_siguiente1, ngram)
            
            if selected_ngram is None:
                break
            
            # Agregar el siguiente término a la oración y actualizar el contexto
            if ngram == 3:
                new_word = selected_ngram[2]
                sentence.extend([selected_ngram[1], selected_ngram[2]])
                context = selected_ngram[2]
            elif ngram == 2:
                new_word = selected_ngram[1]
                sentence.append(new_word)
                context = selected_ngram[1]

            if ngram == 3:
                if selected_ngram[2] == '</s>':
                    break
            elif ngram == 2:
                if selected_ngram[1] == '</s>':
                    break

            
        
        sentence = sentence[:-1]
        return ' '.join(sentence)

# Crear ventana principal
class AplicacionModelosLenguaje:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplicación de Modelos de Lenguaje")

        self.tokens = []
        self.modelos_lenguaje = {} 
        self.tipo_modelo = {}  

        # Crear un notebook (pestañas)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill="both")

        # Crear pestañas
        self.crear_pestañas()

    def crear_pestañas(self):
        # Pestaña para crear modelos de lenguaje
        self.tab_modelo_lenguaje = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_modelo_lenguaje, text="Create language models")
        self.crear_modelo_lenguaje(self.tab_modelo_lenguaje)

        # Pestaña para texto predictivo
        self.tab_texto_predictivo = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_texto_predictivo, text="Predictive text")
        self.abrir_texto_predictivo(self.tab_texto_predictivo)

        # Pestaña para generación de texto
        self.tab_generacion_texto = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_generacion_texto, text="Text generation")
        self.abrir_generacion_texto(self.tab_generacion_texto)

        # Pestaña para probabilidad condicional
        self.tab_probabilidad_condicional = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_probabilidad_condicional, text="Conditional probability")
        self.abrir_probabilidad_condicional(self.tab_probabilidad_condicional)

    # Función para mostrar la interfaz de Crear Modelos de Lenguaje
    def crear_modelo_lenguaje(self, parent):
        marco_modelo = tk.Frame(parent)
        marco_modelo.pack(pady=20)

        # Etiqueta "Cargar corpus"
        label_cargar = tk.Label(marco_modelo, text="Load corpus", fg="black", font=("Arial", 10, "bold"))
        label_cargar.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        # Barra de búsqueda (decorativa por ahora)
        entry_busqueda = tk.Entry(marco_modelo, width=40)
        entry_busqueda.grid(row=0, column=1, padx=10, pady=10)

        # Botón "Browse" que permite buscar el archivo
        boton_browse = tk.Button(marco_modelo, text="Browse", command=self.cargar_corpus)
        boton_browse.grid(row=0, column=2, padx=10, pady=10)

        # Guardar la referencia del cuadro de texto en la instancia de la clase
        self.entry_busqueda = tk.Entry(marco_modelo, width=40)
        self.entry_busqueda.grid(row=0, column=1, padx=10, pady=10)

        # Botones para generar bigramas y trigramas
        boton_bigramas = tk.Button(marco_modelo, text="Generate bigrams", command=self.generar_bigramas)
        boton_bigramas.grid(row=2, column=0, padx=10, pady=10)

        boton_trigramas = tk.Button(marco_modelo, text="Generate trigrams", command=self.generar_trigramas)
        boton_trigramas.grid(row=2, column=1, padx=10, pady=10)

    # Función para mostrar la interfaz de Texto Predictivo
    def abrir_texto_predictivo(self, parent):
        marco_prediccion = tk.Frame(parent)
        marco_prediccion.pack(pady=20)

        # Etiqueta "Load language model"
        label_cargar_modelo = tk.Label(marco_prediccion, text="Load language model", fg="black", font=("Arial", 10, "bold"))
        label_cargar_modelo.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        # Cuadro de búsqueda
        self.entry_busqueda2 = tk.Entry(marco_prediccion, width=40)
        self.entry_busqueda2.grid(row=0, column=1, padx=10, pady=10)

        # Botón "Browse"
        boton_browse = tk.Button(marco_prediccion, text="Browse", command=self.cargar_modelos)
        boton_browse.grid(row=0, column=2, padx=10, pady=5)

        # Etiqueta "Write a word (or two words to start a sentence)"
        label_write_word = tk.Label(marco_prediccion, text="Write a word (or two words to start a sentence)", fg="black", font=("Arial", 10, "bold"))
        label_write_word.grid(row=1, column=0, padx=10, pady=5, columnspan=2, sticky="w")

        # Entrada para escribir una palabra o dos
        self.entry_word = tk.Entry(marco_prediccion, width=30)
        self.entry_word.grid(row=2, column=0, padx=10, pady=5)

        # Botón "Next word"
        boton_next_word = tk.Button(marco_prediccion, text="Next word", command=self.predecir_palabra)
        boton_next_word.grid(row=2, column=1, padx=10, pady=5)

        # Lista desplegable para seleccionar la cantidad de palabras más probables
        opciones = ["3 most probable words", "5 most probable words", "10 most probable words"]
        self.combo_opciones = ttk.Combobox(marco_prediccion, values=opciones, state="readonly")
        self.combo_opciones.current(0)  # Por defecto seleccionamos la opción "3 most probable words"
        self.combo_opciones.grid(row=2, column=2, padx=10, pady=5)

        # Botón "Add word"
        boton_add_word = tk.Button(marco_prediccion, text="Add word", command=self.agregar_palabra)
        boton_add_word.grid(row=3, column=2, padx=10, pady=5)

        # Etiqueta "Generated text"
        label_generated_text = tk.Label(marco_prediccion, text="Generated text", fg="black", font=("Arial", 10, "bold"))
        label_generated_text.grid(row=4, column=0, padx=10, pady=5, sticky="w")
        # Label para mostrar el texto generado
        self.label_generated_text = tk.Label(
            marco_prediccion, 
            text="", 
            fg="black", 
            bg="white",  # Establece el fondo en blanco
            font=("Arial", 10), 
            wraplength=400, 
            justify="left"      
        )
        self.label_generated_text.grid(row=5, column=0, columnspan=3, padx=10, pady=5)

    # Función para mostrar la interfaz de Generación de Texto
    def abrir_generacion_texto(self, parent):
        marco_generacion = tk.Frame(parent)
        marco_generacion.pack(pady=20)

        # Etiqueta "Load corpus" en negritas
        label_load_corpus = tk.Label(marco_generacion, text="Load corpus", font=("Arial", 10, "bold"))
        label_load_corpus.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        # Cuadro de búsqueda (decorativo por ahora)
        self.entry_busqueda3 = tk.Entry(marco_generacion, width=40)
        self.entry_busqueda3.grid(row=0, column=1, padx=10, pady=10)

        # Botón "Browse"
        boton_browse = tk.Button(marco_generacion, text="Browse", command=self.cargar_modelo)
        boton_browse.grid(row=0, column=2, padx=10, pady=5)

        # Botón "Generate sentence"
        boton_generar_oracion = tk.Button(marco_generacion, text="Generate sentence", command=self.generar_oracion)
        boton_generar_oracion.grid(row=1, column=0, padx=10, pady=5, columnspan=2)

        # Etiqueta "Generated text" en negritas
        label_generated_text = tk.Label(marco_generacion, text="Generated text", font=("Arial", 10, "bold"))
        label_generated_text.grid(row=2, column=0, padx=10, pady=5, sticky="w")

        # Cuadro de texto para mostrar el texto generado
        self.text_generated = tk.Text(marco_generacion, height=10, width=60)
        self.text_generated.grid(row=3, column=0, columnspan=3, padx=10, pady=5)

    # Función para mostrar la interfaz de Probabilidad Condicional
    def abrir_probabilidad_condicional(self, parent):
        marco_probabilidad = tk.Frame(parent)
        marco_probabilidad.pack(pady=20)

        # Etiqueta "Load language model" en negritas
        label_cargar_modelo = tk.Label(marco_probabilidad, text="Load language model", font=("Arial", 10, "bold"))
        label_cargar_modelo.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        # Cuadro de búsqueda
        self.entry_busqueda4 = tk.Entry(marco_probabilidad, width=30)
        self.entry_busqueda4.grid(row=0, column=1, padx=10, pady=5)

        # Botones "Browse"
        boton_browse = tk.Button(marco_probabilidad, text="Browse", command=self.cargar_modelo_probabilidad)
        boton_browse.grid(row=0, column=2, padx=10, pady=5)

        # Lista de modelos
        self.lista_modelos = tk.Listbox(marco_probabilidad, height=5)
        self.lista_modelos.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        # Etiqueta "Test sentence" en negritas
        label_test_sentence = tk.Label(marco_probabilidad, text="Test sentence", font=("Arial", 10, "bold"))
        label_test_sentence.grid(row=2, column=0, padx=10, pady=5, sticky="w")

        # Entrada para escribir la oración de prueba
        self.entry_sentence = tk.Entry(marco_probabilidad, width=30)
        self.entry_sentence.grid(row=2, column=1, padx=10, pady=5)

        # Botón "Determine joint probability"
        boton_joint_prob = tk.Button(marco_probabilidad, text="Determine joint probability", command=self.calcular_probabilidad)
        boton_joint_prob.grid(row=2, column=2, padx=10, pady=5)

        # Etiqueta "Results" en negritas
        label_results = tk.Label(marco_probabilidad, text="Results", font=("Arial", 10, "bold"))
        label_results.grid(row=3, column=0, padx=10, pady=5, sticky="w")

        # Tabla de resultados (modelos y probabilidad conjunta)
        self.tabla_resultados = ttk.Treeview(marco_probabilidad, columns=("Modelo", "Probabilidad conjunta"), show="headings")
        self.tabla_resultados.heading("Modelo", text="Language model")
        self.tabla_resultados.heading("Probabilidad conjunta", text="Joint probability")
        self.tabla_resultados.grid(row=4, column=0, columnspan=3, padx=10, pady=5)

    def cargar_corpus(self):
        archivo = filedialog.askopenfilename(filetypes=[("Archivos de texto", "*.txt")])
        if archivo:
            nombre_archivo = os.path.basename(archivo)  # Obtener solo el nombre del archivo
            self.entry_busqueda.delete(0, tk.END)  # Limpiar el cuadro de texto
            self.entry_busqueda.insert(0, nombre_archivo)  # Insertar el nombre del archivo
            print(f"Archivo cargado: {nombre_archivo}")

            # Leer el archivo y procesar los tokens
            with open(archivo, 'r', encoding='utf-8') as f:
                lineas = f.readlines()
                corpus = [linea.strip().split() for linea in lineas if linea.strip()]
                self.tokens = [token for sublist in corpus for token in sublist]

    # Funciones mock para cada acción
    def generar_bigramas(self):
        if self.tokens:
            nombre_archivo_salida = "bigramas.csv"
            self.crear_bigramas(self.tokens, nombre_archivo_salida)
        else:
            print("Primero debes cargar un archivo.")

    def generar_trigramas(self):
        if self.tokens:
            nombre_archivo_salida = "trigramas.csv"
            self.crear_trigramas(self.tokens, nombre_archivo_salida)
        else:
            print("Primero debes cargar un archivo.")

    def crear_bigramas(self, tokens, nombre_archivo_salida):
        """
        Función que genera un archivo CSV con bigramas, sus frecuencias y probabilidades condicionales.
        """
        ruta_salida = os.path.join(os.getcwd(), nombre_archivo_salida)

        # Extraer bigramas
        bigramas = list(zip(tokens, islice(tokens, 1, None)))

        # Contar frecuencias de bigramas y unigrama (término único)
        conteo_bigramas = Counter(bigramas)
        conteo_unigramas = Counter(tokens)

        # Calcular las probabilidades condicionales para bigramas
        datos_bigramas = [
            (termino1, termino2, freq, conteo_unigramas[termino1], freq / conteo_unigramas[termino1])
            for (termino1, termino2), freq in conteo_bigramas.items()
        ]

        # Ordenar los datos por frecuencia en orden descendente
        datos_bigramas.sort(key=lambda x: x[2], reverse=True)

        # Escribir los datos de bigramas en un archivo CSV
        with open(ruta_salida, 'w', newline='', encoding='utf-8') as csvfile:
            escritor = csv.writer(csvfile)
            escritor.writerow(['Term 1', 'Term 2', 'frequency of bigram', 'frequency of context (Term 1)', ' conditional probability of bigram'])
            escritor.writerows(datos_bigramas)

        print(f"Archivo de bigramas generado: {ruta_salida}")

    def crear_trigramas(self, tokens, nombre_archivo_salida):
        """
        Función que genera un archivo CSV con trigramas, sus frecuencias y probabilidades condicionales.
        """
        ruta_salida = os.path.join(os.getcwd(), nombre_archivo_salida)

        # Extraer trigramas
        trigramas = list(zip(tokens, islice(tokens, 1, None), islice(tokens, 2, None)))

        # Contar frecuencias de trigramas y bigramas (contexto del trigramas)
        conteo_trigramas = Counter(trigramas)
        conteo_bigramas = Counter(list(zip(tokens, islice(tokens, 1, None))))

        # Calcular las probabilidades condicionales para trigramas
        datos_trigramas = [
            (termino1, termino2, termino3, freq, conteo_bigramas[(termino1, termino2)], freq / conteo_bigramas[(termino1, termino2)])
            for (termino1, termino2, termino3), freq in conteo_trigramas.items()
        ]

        # Ordenar los datos por frecuencia en orden descendente
        datos_trigramas.sort(key=lambda x: x[3], reverse=True)

        # Escribir los datos de trigramas en un archivo CSV
        with open(ruta_salida, 'w', newline='', encoding='utf-8') as csvfile:
            escritor = csv.writer(csvfile)
            escritor.writerow(['Term 1', 'Term 2', 'Term 3', 'frequency of trigram', 'frequency of context (bigram Term1 + Term2)', 'probability of trigram'])
            escritor.writerows(datos_trigramas)

        print(f"Archivo de trigramas generado: {ruta_salida}")

    def cargar_modelo(self):
        global ngrams_global
        archivo = filedialog.askopenfilename(filetypes=[("Archivos CSV", "*.csv")])
        if archivo:
            nombre_archivo = os.path.basename(archivo)
            self.entry_busqueda3.delete(0, tk.END)
            self.entry_busqueda3.insert(0, nombre_archivo)
            print(f"Modelo de lenguaje cargado: {nombre_archivo}")
            with open(archivo, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    ngrams_global.append(row)
    
    def cargar_modelos(self):
        global ngrams_global
        archivo = filedialog.askopenfilename(filetypes=[("Archivos CSV", "*.csv")])
        if archivo:
            nombre_archivo = os.path.basename(archivo)
            self.entry_busqueda2.delete(0, tk.END)
            self.entry_busqueda2.insert(0, nombre_archivo)
            print(f"Modelo de lenguaje cargado: {nombre_archivo}")
            with open(archivo, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    ngrams_global.append(row)

    def generar_oracion(self):
        ngram = 2 
        #Generar oración
        sentence = generate_sentence(ngram, ngrams_global)
        print(f"Oración generada: {sentence}")
        
        self.text_generated.delete(1.0, tk.END)
        self.text_generated.insert(tk.END, sentence)
        
    def predecir_palabra(self):
        """Predice las siguientes palabras más probables cuando se presiona 'Next word'"""
        global ngrams_global

        # Obtener el texto inicial
        entrada = self.entry_word.get().strip()
        
        if not entrada:
            messagebox.showerror("Error", "Por favor ingrese una o dos palabras para comenzar")
            return
        
        # Determinar si estamos trabajando con bigramas o trigramas
        es_bigrama = len(ngrams_global[0]) == 5
        palabras = entrada.split()
        predicciones = []
        
        # Obtener las predicciones según el tipo de n-grama
        if es_bigrama:
            contexto = palabras[-1]
            for fila in ngrams_global[1:]:  # Saltamos la cabecera
                if fila[0] == contexto:
                    predicciones.append((fila[1], float(fila[4])))
        else:  # trigrama
            if len(palabras) < 2:
                messagebox.showwarning("Advertencia", "Para trigramas, ingrese al menos dos palabras")
                return
            
            contexto1, contexto2 = palabras[-2], palabras[-1]
            for fila in ngrams_global[1:]:
                if fila[0] == contexto1 and fila[1] == contexto2:
                    predicciones.append((fila[2], float(fila[5])))
        
        # Ordenar por probabilidad y obtener las 3 más probables
        predicciones.sort(key=lambda x: x[1], reverse=True)
        
        # Preparar las opciones para el combobox
        opciones = ["", "", ""]  # Inicializar con strings vacíos
        for i, (palabra, _) in enumerate(predicciones[:3]):
            if palabra == '</s>':
                opciones[i] = '.'
            else:
                opciones[i] = palabra
        
        # Siempre añadir el punto como opción
        opciones.append('.')
        
        # Actualizar el combobox
        self.combo_opciones['values'] = opciones
        if opciones:
            self.combo_opciones.current(0)

    def cargar_modelo_probabilidad(self):
        archivo = filedialog.askopenfilename(filetypes=[("Archivos CSV", "*.csv")])
        if archivo:
            nombre_archivo = os.path.basename(archivo)
            self.entry_busqueda4.delete(0, tk.END)
            self.entry_busqueda4.insert(0, nombre_archivo)
            print(f"Archivo cargado: {nombre_archivo}")

            # Cargar el modelo desde el archivo CSV
            self.leer_modelo_csv(archivo, nombre_archivo)

    def leer_modelo_csv(self, archivo, nombre_archivo):
        modelo = defaultdict(float)
        frecuencia_terminos = defaultdict(int)  # Diccionario para frecuencias individuales y de bigramas/trigramas

        with open(archivo, 'r', encoding='utf-8') as csvfile:
            lector_csv = csv.reader(csvfile)
            encabezados = next(lector_csv)  # Leer los encabezados para determinar el tipo de modelo

            if len(encabezados) == 5:  # Es un modelo de bigramas
                self.tipo_modelo[nombre_archivo] = "bigram"
                for fila in lector_csv:
                    termino1, termino2, freq_bigrama, freq_termino1, _ = fila
                    modelo[(termino1, termino2)] = int(freq_bigrama)  # Frecuencia del bigrama
                    frecuencia_terminos[termino1] = int(freq_termino1)  # Frecuencia del término individual

            elif len(encabezados) == 6:  # Es un modelo de trigramas
                self.tipo_modelo[nombre_archivo] = "trigram"
                for fila in lector_csv:
                    termino1, termino2, termino3, freq_trigrama, freq_bigrama, _ = fila
                    modelo[(termino1, termino2, termino3)] = int(freq_trigrama)  # Frecuencia del trigrama
                    frecuencia_terminos[(termino1, termino2)] = int(freq_bigrama)  # Frecuencia del bigrama

        self.modelos_lenguaje[nombre_archivo] = (modelo, frecuencia_terminos)
        self.lista_modelos.insert(tk.END, nombre_archivo)
        print(f"Modelo {nombre_archivo} agregado con éxito.")

    def calcular_probabilidad(self):
        oracion = ["<s>"] + self.entry_sentence.get().split() + ["</s>"]
        if not oracion:
            print("Por favor, ingresa una oración.")
            return

        # Calcular la probabilidad conjunta para cada modelo de lenguaje
        resultados = []
        for nombre_modelo, modelo in self.modelos_lenguaje.items():
            probabilidad = self.calcular_probabilidad_conjunta(oracion, modelo, self.tipo_modelo[nombre_modelo])
            resultados.append((nombre_modelo, probabilidad))
        
        # Ordenar los resultados en orden descendente de probabilidad conjunta
        resultados_ordenados = sorted(resultados, key=lambda x: x[1], reverse=True)
        
        # Limpiar la tabla y mostrar los resultados ordenados
        self.tabla_resultados.delete(*self.tabla_resultados.get_children())
        for nombre_modelo, probabilidad in resultados_ordenados:
            self.tabla_resultados.insert("", "end", values=(nombre_modelo, f"{probabilidad:.18f}"))

    def calcular_probabilidad_conjunta(self, oracion, modelo_data, tipo_modelo):
        modelo, frecuencia_terminos = modelo_data
        probabilidad_conjunta = 1.0

        # Construir vocabulario para el suavizado de Laplace
        vocabulario = set()
        if tipo_modelo == "bigram":
            for (termino1, termino2) in modelo.keys():
                vocabulario.update([termino1, termino2])
        elif tipo_modelo == "trigram":
            for (termino1, termino2, termino3) in modelo.keys():
                vocabulario.update([termino1, termino2, termino3])
        V = len(vocabulario)

        # Cálculo de probabilidad conjunta
        if len(oracion) == 1:  # Unigrama
            palabra = oracion[0]
            freq_palabra = frecuencia_terminos.get(palabra, 0)
            total_frecuencias = sum(frecuencia_terminos.values())
            
            # Usar probabilidad directa si existe, de lo contrario aplicar suavizado de Laplace
            probabilidad_conjunta = (freq_palabra / total_frecuencias) if freq_palabra > 0 else (1 / (total_frecuencias + V))
            print(w1, w2, freq_palabra, total_frecuencias, probabilidad_conjunta, V)

        elif tipo_modelo == "bigram" or (tipo_modelo == "trigram" and len(oracion) == 2):  # Bigramas
            for w1, w2 in zip(oracion, oracion[1:]):
                if tipo_modelo == "trigram":
                    # En el caso de trigramas, accedemos a la frecuencia del contexto como bigrama en el modelo trigramal
                    freq_bigrama = frecuencia_terminos.get((w1, w2), 0)  # Frecuencia del contexto bigrama
                else:
                    # En el caso de bigramas, accedemos directamente al bigrama en el modelo
                    freq_bigrama = modelo.get((w1, w2), 0)
                    
                freq_termino1 = frecuencia_terminos.get(w1, 0)

                # Usar probabilidad directa si existe, de lo contrario aplicar suavizado de Laplace
                if freq_bigrama > 0 and freq_termino1 > 0:
                    probabilidad_condicional = freq_bigrama / freq_termino1
                else:
                    probabilidad_condicional = 1 / (freq_termino1 + V)
                probabilidad_conjunta *= probabilidad_condicional
                print(w1, w2, freq_bigrama, freq_termino1, probabilidad_condicional, V)

        elif tipo_modelo == "trigram" and len(oracion) >= 3:  # Trigramas
            for w1, w2, w3 in zip(oracion, oracion[1:], oracion[2:]):
                freq_trigrama = modelo.get((w1, w2, w3), 0)
                freq_bigrama = frecuencia_terminos.get((w1, w2), 0)

                # Usar probabilidad directa si existe, de lo contrario aplicar suavizado de Laplace
                if freq_trigrama > 0 and freq_bigrama > 0:
                    probabilidad_condicional = freq_trigrama / freq_bigrama
                else:
                    probabilidad_condicional = 1 / (freq_bigrama + V)
                probabilidad_conjunta *= probabilidad_condicional
                print(w1, w2, w3, freq_trigrama, freq_bigrama, probabilidad_condicional, V)

        return probabilidad_conjunta
    
    def agregar_palabra(self):
        """Agrega la palabra seleccionada al texto sin borrar el contenido actual, incluyendo el mensaje inicial del usuario"""
    
    # Extraer el mensaje inicial ingresado por el usuario (si no se ha extraído ya)
        if not self.label_generated_text.cget("text"):  # Si el label está vacío
            mensaje_inicial = self.entry_word.get().strip()
            self.label_generated_text.config(text=mensaje_inicial)

        palabra_seleccionada = self.combo_opciones.get()
        if not palabra_seleccionada:
            return  # No se hace nada si no hay una palabra seleccionada
    
    # Obtener el texto actual del label (mensaje inicial + palabras seleccionadas)
        texto_actual = self.label_generated_text.cget("text").strip()
    
        if palabra_seleccionada == '.':
        # Si se selecciona el punto, terminar la oración
            nuevo_texto = f"{texto_actual}.".strip()
            self.label_generated_text.config(text=nuevo_texto)
            self.entry_word.delete(0, tk.END)
            self.combo_opciones.set('')  # Limpiar el combobox
            messagebox.showinfo("Información", "Oración finalizada. Puede comenzar una nueva.")
        else:
            # Concatenar la palabra seleccionada al texto actual
            nuevo_texto = f"{texto_actual} {palabra_seleccionada}".strip()
            self.label_generated_text.config(text=nuevo_texto)
            
            # Actualizar el contexto para la siguiente predicción
            palabras = nuevo_texto.split()
            es_bigrama = len(ngrams_global[0]) == 5
            
            # Actualizar entry_word con el nuevo contexto
            if es_bigrama:
                self.entry_word.delete(0, tk.END)
                self.entry_word.insert(0, palabra_seleccionada)
            else:
                if len(palabras) >= 2:
                    self.entry_word.delete(0, tk.END)
                    self.entry_word.insert(0, f"{palabras[-2]} {palabra_seleccionada}")
                else:
                    self.entry_word.delete(0, tk.END)
                    self.entry_word.insert(0, palabra_seleccionada)

# Ejecutar la aplicación
root = tk.Tk()
app = AplicacionModelosLenguaje(root)
root.mainloop()
