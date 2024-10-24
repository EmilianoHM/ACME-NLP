import os
import re

# Función para filtrar los mensajes de "Emiliano H M"
def filtrar_mensajes(texto):
    mensajes = []
    # Expresión regular para buscar "Usuario" y capturar todo después de ":"
    # patron = re.compile(r"Usuario: (.+)") # Cambiar usuario por el nombre de Whats
    # Que el usuario sea Rolón
    patron = re.compile(r"Rolón: (.+)")
    print("[DEBUG] Iniciando el filtrado de mensajes...")
    
    
    for linea in texto.splitlines():
        match = patron.search(linea)
        if match:
            mensaje = match.group(0)
            # Verificar si el mensaje contiene "omitido" u "omitida" y no incluirlo si es así
            if "omitido" not in mensaje.lower() and "omitida" not in mensaje.lower():
                mensajes.append(mensaje)
                print(f"[DEBUG] Mensaje encontrado: {mensaje}")
            else:
                print(f"[DEBUG] Mensaje omitido: {mensaje}")
    
    if not mensajes:
        print("[DEBUG] No se encontraron mensajes de Emiliano H M en este archivo.")
        
    return "\n".join(mensajes)

# Directorio donde está el script
directorio = os.path.dirname(os.path.abspath(__file__))
print(f"[DEBUG] Directorio del script: {directorio}")

# Archivo de salida
archivo_salida = "mensajes_usuario.txt"

# Abrir archivo de salida en modo escritura
with open(os.path.join(directorio, archivo_salida), "w", encoding="utf-8") as salida:
    print(f"[DEBUG] Archivo de salida: {archivo_salida}")
    
    # Iterar sobre los archivos en el directorio
    for archivo_nombre in os.listdir(directorio):
        if archivo_nombre.endswith(".txt") and archivo_nombre != archivo_salida:
            print(f"[DEBUG] Procesando archivo: {archivo_nombre}")
            
            # Leer contenido del archivo
            with open(os.path.join(directorio, archivo_nombre), "r", encoding="utf-8") as archivo:
                contenido = archivo.read()
                
                # Verificar si el archivo tiene contenido
                if contenido.strip():
                    print("[DEBUG] El archivo tiene contenido, comenzando el filtrado...")
                    # Filtrar mensajes de "Emiliano H M"
                    mensajes_filtrados = filtrar_mensajes(contenido)
                    
                    # Escribir los mensajes en el archivo de salida si se encontraron
                    if mensajes_filtrados:
                        salida.write(mensajes_filtrados + "\n")
                        print(f"[DEBUG] Mensajes filtrados guardados en el archivo de salida.")
                    else:
                        print(f"[DEBUG] No se encontraron mensajes de Emiliano H M en {archivo_nombre}.")
                else:
                    print(f"[DEBUG] El archivo {archivo_nombre} está vacío.")
                    
print("Mensajes filtrados y guardados en", archivo_salida)
