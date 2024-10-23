import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import os

# Crear ventana principal
class AplicacionModelosLenguaje:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplicación de Modelos de Lenguaje")

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
        boton_browse = tk.Button(marco_modelo, text="Browse", command=self.cargar_archivo)
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
        self.entry_busqueda = tk.Entry(marco_prediccion, width=40)
        self.entry_busqueda.grid(row=0, column=1, padx=10, pady=10)

        # Botón "Browse"
        boton_browse = tk.Button(marco_prediccion, text="Browse", command=self.cargar_archivo)
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

        # Cuadro de texto para mostrar el texto generado
        self.text_generated = tk.Text(marco_prediccion, height=10, width=60)
        self.text_generated.grid(row=5, column=0, columnspan=3, padx=10, pady=5)

    # Función para mostrar la interfaz de Generación de Texto
    def abrir_generacion_texto(self, parent):
        marco_generacion = tk.Frame(parent)
        marco_generacion.pack(pady=20)

        # Etiqueta "Load corpus" en negritas
        label_load_corpus = tk.Label(marco_generacion, text="Load corpus", font=("Arial", 10, "bold"))
        label_load_corpus.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        # Cuadro de búsqueda (decorativo por ahora)
        self.entry_busqueda = tk.Entry(marco_generacion, width=40)
        self.entry_busqueda.grid(row=0, column=1, padx=10, pady=10)

        # Botón "Browse"
        boton_browse = tk.Button(marco_generacion, text="Browse", command=self.cargar_archivo)
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
        self.entry_busqueda = tk.Entry(marco_probabilidad, width=30)
        self.entry_busqueda.grid(row=0, column=1, padx=10, pady=5)

        # Botones "Browse" y "Add model"
        boton_browse = tk.Button(marco_probabilidad, text="Browse", command=self.cargar_archivo)
        boton_browse.grid(row=0, column=2, padx=10, pady=5)

        boton_add_model = tk.Button(marco_probabilidad, text="Add model", command=self.agregar_modelo)
        boton_add_model.grid(row=0, column=3, padx=10, pady=5)

        # Lista de modelos
        self.lista_modelos = tk.Listbox(marco_probabilidad, height=5)
        self.lista_modelos.insert(1, "Model 1")
        self.lista_modelos.insert(2, "Model 2")
        self.lista_modelos.insert(3, "...")
        self.lista_modelos.insert(4, "Model n")
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
        self.tabla_resultados.insert("", "end", values=("Model 1", "0.05"))
        self.tabla_resultados.insert("", "end", values=("Model 2", "0.04"))
        self.tabla_resultados.insert("", "end", values=("...", "0.001"))
        self.tabla_resultados.insert("", "end", values=("Model n", "0.0001"))
        self.tabla_resultados.grid(row=4, column=0, columnspan=3, padx=10, pady=5)

    def cargar_archivo(self):
        archivo = filedialog.askopenfilename()
        if archivo:
            nombre_archivo = os.path.basename(archivo)  # Obtener solo el nombre del archivo
            self.entry_busqueda.delete(0, tk.END)  # Limpiar el cuadro de texto
            self.entry_busqueda.insert(0, nombre_archivo)  # Insertar el nombre del archivo
            print(f"Archivo cargado: {nombre_archivo}")

    # Funciones mock para cada acción
    def generar_bigramas(self):
        print("Generar bigramas...")

    def generar_trigramas(self):
        print("Generar trigramas...")

    def cargar_modelo(self):
        print("Cargar modelo de lenguaje...")

    def predecir_palabra(self):
        palabra_inicial = self.entry_word.get()
        print(f"Prediciendo palabra siguiente para: {palabra_inicial}")

    def agregar_palabra(self):
        print("Palabra añadida")

    def generar_oracion(self):
        print("Generar oración...")

    def agregar_modelo(self):
        print("Modelo agregado")

    def calcular_probabilidad(self):
        print("Calculando probabilidad conjunta...")

# Ejecutar la aplicación
root = tk.Tk()
app = AplicacionModelosLenguaje(root)
root.mainloop()
