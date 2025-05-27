import pygame
import random
import csv
import joblib  # Para cargar modelos
import os

# Inicializar Pygame
pygame.init()

# Dimensiones de la pantalla
w, h = 800, 400
pantalla = pygame.display.set_mode((w, h))
pygame.display.set_caption("Juego: Disparo de Bala, Salto, Nave y Menú")

# Colores
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)

# Variables del jugador, bala, nave, fondo, etc.
jugador = None
bala = None
fondo = None
nave = None
menu = None

# Variables de salto
salto = False
salto_altura = 15
gravedad = 1
en_suelo = True

# Variables de pausa y menú
pausa = False
fuente = pygame.font.SysFont('Arial', 24)
menu_activo = True
modo_auto = False
modelo_seleccionado = ""
modelo_clasificador = None

# Lista para guardar los datos de velocidad, distancia y salto (target)
datos_modelo = []

# Cargar las imágenes con manejo de errores
try:
    jugador_frames = [
        pygame.image.load('assets/sprites/mono_frame_1.png'),
        pygame.image.load('assets/sprites/mono_frame_2.png'),
        pygame.image.load('assets/sprites/mono_frame_3.png'),
        pygame.image.load('assets/sprites/mono_frame_4.png')
    ]
    bala_img = pygame.image.load('assets/sprites/purple_ball.png')
    fondo_img = pygame.image.load('assets/game/fondo2.png')
    nave_img = pygame.image.load('assets/game/ufo.png')
    menu_img = pygame.image.load('assets/game/menu.png')
except pygame.error as e:
    print("Error cargando imagen:", e)
    pygame.quit()
    exit()

fondo_img = pygame.transform.scale(fondo_img, (w, h))

jugador = pygame.Rect(50, h - 100, 32, 48)
bala = pygame.Rect(w - 50, h - 90, 16, 16)
nave = pygame.Rect(w - 100, h - 100, 64, 64)
menu_rect = pygame.Rect(w // 2 - 135, h // 2 - 90, 270, 180)

current_frame = 0
frame_speed = 10
frame_count = 0

velocidad_bala = -10
bala_disparada = False

fondo_x1 = 0
fondo_x2 = w

#Regresion

def entrenar_modelo_regresion():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import pandas as pd

    try:
        df = pd.read_csv("datos_modelo.csv")
        X = df[["velocidad_bala", "distancia"]]
        y = df["salto_hecho"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        modelo = LogisticRegression()
        modelo.fit(X_train, y_train)

        # Evaluar
        y_pred = modelo.predict(X_test)
        precision = accuracy_score(y_test, y_pred)
        print(f"[Entrenamiento Automático] Precisión: {precision:.2f}")

        os.makedirs("modelos", exist_ok=True)
        joblib.dump(modelo, "modelos/regresion_lineal.pkl")
        print("Modelo actualizado: modelos/regresion_lineal.pkl")

    except Exception as e:
        print("Error al entrenar modelo:", e)

#arbol

def entrenar_modelo_arbol():
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import joblib

    try:
        df = pd.read_csv("datos_modelo.csv")
        X = df[["velocidad_bala", "distancia"]]
        y = df["salto_hecho"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        modelo = DecisionTreeClassifier()
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"[Árbol] Precisión: {acc:.2f}")

        joblib.dump(modelo, "modelos/decision_tree.pkl")
        print("Modelo guardado: modelos/decision_tree.pkl")

    except Exception as e:
        print("Error al entrenar árbol de decisión:", e)

#Neuronal

def entrenar_modelo_red_neuronal():
    import pandas as pd
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import joblib

    try:
        df = pd.read_csv("datos_modelo.csv")
        X = df[["velocidad_bala", "distancia"]]
        y = df["salto_hecho"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        modelo = MLPClassifier(max_iter=500)
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"[Red Neuronal] Precisión: {acc:.2f}")

        joblib.dump(modelo, "modelos/red_neuronal.pkl")
        print("Modelo guardado: modelos/red_neuronal.pkl")

    except Exception as e:
        print("Error al entrenar red neuronal:", e)

#K-Nearest

def entrenar_modelo_knn():
    import pandas as pd
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import joblib

    try:
        df = pd.read_csv("datos_modelo.csv")
        X = df[["velocidad_bala", "distancia"]]
        y = df["salto_hecho"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        modelo = KNeighborsClassifier(n_neighbors=3)
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"[KNN] Precisión: {acc:.2f}")

        joblib.dump(modelo, "modelos/knn.pkl")
        print("Modelo guardado: modelos/knn.pkl")

    except Exception as e:
        print("Error al entrenar KNN:", e)


def disparar_bala():
    global bala_disparada, velocidad_bala
    if not bala_disparada:
        velocidad_bala = random.randint(-8, -3)
        bala_disparada = True

def reset_bala():
    global bala, bala_disparada
    bala.x = w - 50
    bala_disparada = False

def manejar_salto():
    global jugador, salto, salto_altura, gravedad, en_suelo
    if salto:
        if jugador.y - salto_altura <= h - 100:
            jugador.y -= salto_altura
            salto_altura -= gravedad
        else:
            jugador.y = h - 100
            salto = False
            salto_altura = 15
            en_suelo = True

def guardar_datos():
    global jugador, bala, velocidad_bala, salto
    distancia = abs(jugador.x - bala.x)
    salto_hecho = 1 if salto else 0
    datos_modelo.append((velocidad_bala, distancia, salto_hecho))

def guardar_csv(nombre="datos_modelo.csv"):
    with open(nombre, mode='w', newline='') as archivo:
        writer = csv.writer(archivo)
        writer.writerow(["velocidad_bala", "distancia", "salto_hecho"])
        writer.writerows(datos_modelo)

def pausa_juego():
    global pausa
    pausa = not pausa
    if pausa:
        print("Juego pausado. Datos registrados hasta ahora:", datos_modelo)
    else:
        print("Juego reanudado.")

def mostrar_datos():
    texto = fuente.render(f"Velocidad: {velocidad_bala}, Distancia: {abs(jugador.x - bala.x)}", True, BLANCO)
    pantalla.blit(texto, (10, 10))

def cargar_modelo(nombre_archivo):
    ruta = os.path.join("modelos", nombre_archivo)
    if os.path.exists(ruta):
        return joblib.load(ruta)
    else:
        print(f"Modelo no encontrado: {ruta}")
        return None

def mostrar_menu_modelos():
    global modelo_seleccionado, modelo_clasificador
    pantalla.fill(NEGRO)
    opciones = ["1. Regresión Lineal", "2. Decision Trees", "3. Redes Neuronales", "4. K Nearest Neighbor"]
    nombres_archivos = ["regresion_lineal.pkl", "decision_tree.pkl", "red_neuronal.pkl", "knn.pkl"]
    for i, opcion in enumerate(opciones):
        texto = fuente.render(opcion, True, BLANCO)
        pantalla.blit(texto, (w // 4, h // 2 + i * 30))
    pygame.display.flip()

    seleccionando = True
    while seleccionando:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                guardar_csv()
                entrenar_modelo_arbol()
                entrenar_modelo_knn()
                entrenar_modelo_red_neuronal()
                pygame.quit()
                exit()
            if evento.type == pygame.KEYDOWN:
                index = None
                if evento.key == pygame.K_1:
                    index = 0
                elif evento.key == pygame.K_2:
                    index = 1
                elif evento.key == pygame.K_3:
                    index = 2
                elif evento.key == pygame.K_4:
                    index = 3
                if index is not None:
                    modelo_seleccionado = opciones[index]
                    modelo_clasificador = cargar_modelo(nombres_archivos[index])
                    seleccionando = False
    print("Modelo seleccionado:", modelo_seleccionado)

def mostrar_menu():
    global menu_activo, modo_auto
    pantalla.fill(NEGRO)
    texto = fuente.render("Presiona 'A' para Auto, 'M' para Manual, o 'Q' para Salir", True, BLANCO)
    pantalla.blit(texto, (w // 4, h // 2))
    pygame.display.flip()
    while menu_activo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                guardar_csv()
                entrenar_modelo_arbol()
                entrenar_modelo_knn()
                entrenar_modelo_red_neuronal()
                pygame.quit()
                exit()
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_a:
                    modo_auto = True
                    mostrar_menu_modelos()
                    menu_activo = False
                elif evento.key == pygame.K_m:
                    modo_auto = False
                    menu_activo = False
                elif evento.key == pygame.K_q:
                    # print("Juego terminado. Datos recopilados:", datos_modelo)
                    guardar_csv()
                    entrenar_modelo_arbol()
                    entrenar_modelo_knn()
                    entrenar_modelo_red_neuronal()
                    pygame.quit()
                    exit()

def reiniciar_juego():
    global menu_activo, jugador, bala, nave, bala_disparada, salto, en_suelo
    menu_activo = True
    jugador.x, jugador.y = 50, h - 100
    bala.x = w - 50
    nave.x, nave.y = w - 100, h - 100
    bala_disparada = False
    salto = False
    en_suelo = True
    print("Datos recopilados para el modelo:", datos_modelo)
    mostrar_menu()

def update():
    global bala, velocidad_bala, current_frame, frame_count, fondo_x1, fondo_x2

    fondo_x1 -= 1
    fondo_x2 -= 1
    if fondo_x1 <= -w:
        fondo_x1 = w
    if fondo_x2 <= -w:
        fondo_x2 = w

    pantalla.blit(fondo_img, (fondo_x1, 0))
    pantalla.blit(fondo_img, (fondo_x2, 0))

    frame_count += 1
    if frame_count >= frame_speed:
        current_frame = (current_frame + 1) % len(jugador_frames)
        frame_count = 0

    pantalla.blit(jugador_frames[current_frame], (jugador.x, jugador.y))
    pantalla.blit(nave_img, (nave.x, nave.y))

    if bala_disparada:
        bala.x += velocidad_bala

    if bala.x < 0:
        reset_bala()

    pantalla.blit(bala_img, (bala.x, bala.y))

    if jugador.inflate(-10, -10).colliderect(bala):
        print("Colisión detectada!")
        reiniciar_juego()

    mostrar_datos()

def main():
    global salto, en_suelo, bala_disparada
    reloj = pygame.time.Clock()
    mostrar_menu()
    correr = True

    while correr:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                guardar_csv()
                entrenar_modelo_arbol()
                entrenar_modelo_knn()
                entrenar_modelo_red_neuronal()
                correr = False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_SPACE and en_suelo and not pausa:
                    salto = True
                    en_suelo = False
                if evento.key == pygame.K_p:
                    pausa_juego()
                if evento.key == pygame.K_q:
                    # print("Juego terminado. Datos recopilados:", datos_modelo)
                    guardar_csv()
                    entrenar_modelo_arbol()
                    entrenar_modelo_knn()
                    entrenar_modelo_red_neuronal()
                    pygame.quit()
                    exit()

        if not pausa:
            if not modo_auto:
                if salto:
                    manejar_salto()
                guardar_datos()
            else:
                if modelo_clasificador:
                    distancia = abs(jugador.x - bala.x)
                    entrada = pd.DataFrame([[velocidad_bala, distancia]], columns=["velocidad_bala", "distancia"])
                    prediccion = modelo_clasificador.predict(entrada)[0]
                    if prediccion == 1 and en_suelo:
                        salto = True
                        en_suelo = False

            if not bala_disparada:
                disparar_bala()
            update()

        pygame.display.flip()
        reloj.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
