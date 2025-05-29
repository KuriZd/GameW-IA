import pygame
import random
import csv
import joblib

pygame.init()

w, h = 800, 400
pantalla = pygame.display.set_mode((w, h))
pygame.display.set_caption("Juego: Disparo de Bala, Salto, Nave y Menú")

BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)

jugador = None
bala = None
bala2 = None
fondo = None
nave = None
menu = None

salto = False
salto_altura = 15
gravedad = 1
en_suelo = True
velocidad_jugador = 3
posicion_inicial = 50
volver_a_inicio = False

pausa = False
fuente = pygame.font.SysFont('Arial', 24)
menu_activo = True
modo_auto = False
modelo_seleccionado = None

# Datos para entrenamiento
datos_modelo = []

# Cargar imágenes optimizadas
jugador_frames = [
    pygame.image.load('assets/sprites/mono_frame_1.png').convert_alpha(),
    pygame.image.load('assets/sprites/mono_frame_2.png').convert_alpha(),
    pygame.image.load('assets/sprites/mono_frame_3.png').convert_alpha(),
    pygame.image.load('assets/sprites/mono_frame_4.png').convert_alpha()
]

bala_img = pygame.image.load('assets/sprites/purple_ball.png').convert_alpha()
fondo_img = pygame.image.load('assets/game/fondo2.png').convert()
nave_img = pygame.image.load('assets/game/ufo.png').convert_alpha()
menu_img = pygame.image.load('assets/game/menu.png').convert_alpha()
fondo_img = pygame.transform.scale(fondo_img, (w, h))

jugador = pygame.Rect(posicion_inicial, h - 100, 32, 48)
bala = pygame.Rect(w - 50, h - 90, 16, 16)
bala2 = pygame.Rect(50, 0, 16, 16)
nave = pygame.Rect(w - 100, h - 100, 64, 64)
menu_rect = pygame.Rect(w // 2 - 135, h // 2 - 90, 270, 180)

current_frame = 0
frame_speed = 10
frame_count = 0

velocidad_bala = -10
bala_disparada = False
velocidad_bala2 = 9
bala2_disparada = True

fondo_x1 = 0
fondo_x2 = w

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
        jugador.y -= salto_altura
        salto_altura -= gravedad
        if jugador.y >= h - 100:
            jugador.y = h - 100
            salto = False
            salto_altura = 15
            en_suelo = True

def detectar_colisiones():
    return jugador.colliderect(bala) or jugador.colliderect(bala2)

def update():
    global bala, velocidad_bala, current_frame, frame_count, fondo_x1, fondo_x2, bala2

    fondo_x1 -= 1
    fondo_x2 -= 1
    if fondo_x1 <= -w: fondo_x1 = w
    if fondo_x2 <= -w: fondo_x2 = w

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

    if bala2_disparada:
        bala2.y += velocidad_bala2
    if bala2.y > h:
        bala2.y = 0
        bala2.x = 50
    pantalla.blit(bala_img, (bala2.x, bala2.y))

    if detectar_colisiones():
        print("\u00a1Colisi\u00f3n detectada!")
        if not modo_auto:
            reiniciar_juego()

def guardar_datos():
    global jugador, bala, velocidad_bala, salto
    distancia = abs(jugador.x - bala.x)
    salto_hecho = 1 if salto else 0
    datos_modelo.append((velocidad_bala, distancia, salto_hecho))

def guardar_csv():
    with open("datos_modelo.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["velocidad_bala", "distancia", "salto_hecho"])
        writer.writerows(datos_modelo)

def pausa_juego():
    global pausa
    pausa = not pausa
    if pausa:
        print("Juego pausado. Datos registrados hasta ahora:", datos_modelo)
    else:
        print("Juego reanudado.")

def mostrar_menu_modelos():
    global modelo_seleccionado
    pantalla.fill(NEGRO)
    opciones = ["1. Regresión Lineal", "2. Árboles de Decisión", "3. Redes Neuronales", "4. K-Nearest Neighbor"]
    y = h // 2 - 80
    pantalla.blit(fuente.render("Selecciona el modelo automático:", True, BLANCO), (w // 4, y - 40))
    for opcion in opciones:
        pantalla.blit(fuente.render(opcion, True, BLANCO), (w // 4, y))
        y += 30
    pygame.display.flip()

    esperando_seleccion = True
    while esperando_seleccion:
        for evento in pygame.event.get():
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_1:
                    modelo_seleccionado = "Regresión Lineal"
                    esperando_seleccion = False
                elif evento.key == pygame.K_2:
                    modelo_seleccionado = "Árboles de Decisión"
                    esperando_seleccion = False
                elif evento.key == pygame.K_3:
                    modelo_seleccionado = "Redes Neuronales"
                    esperando_seleccion = False
                elif evento.key == pygame.K_4:
                    modelo_seleccionado = "K-Nearest Neighbor"
                    esperando_seleccion = False

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
                    guardar_csv()
                    pygame.quit()
                    exit()

def reiniciar_juego():
    global menu_activo, jugador, bala, nave, bala_disparada, salto, en_suelo, bala2, volver_a_inicio
    menu_activo = True
    jugador.x, jugador.y = posicion_inicial, h - 100
    bala.x = w - 50
    bala2.x, bala2.y = 50, 0
    nave.x, nave.y = w - 100, h - 100
    bala_disparada = False
    salto = False
    en_suelo = True
    volver_a_inicio = False
    print("Datos recopilados para el modelo: ", datos_modelo)
    mostrar_menu()

def main():
    global salto, en_suelo, bala_disparada, volver_a_inicio

    reloj = pygame.time.Clock()
    mostrar_menu()
    correr = True

    while correr:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                guardar_csv()
                correr = False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_UP and en_suelo and not pausa:
                    salto = True
                    en_suelo = False
                if evento.key == pygame.K_p:
                    pausa_juego()
                if evento.key == pygame.K_q:
                    guardar_csv()
                    pygame.quit()
                    exit()

        teclas = pygame.key.get_pressed()
        if teclas[pygame.K_RIGHT] and jugador.x < w - jugador.width and not volver_a_inicio:
            jugador.x += velocidad_jugador + 2
            if jugador.x - posicion_inicial > 30:
                volver_a_inicio = True

        if volver_a_inicio:
            jugador.x -= velocidad_jugador
            if jugador.x <= posicion_inicial:
                jugador.x = posicion_inicial
                volver_a_inicio = False

        if not pausa:
            if not modo_auto:
                if salto:
                    manejar_salto()
                guardar_datos()
            if not bala_disparada:
                disparar_bala()
            update()

        pygame.display.flip()
        reloj.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
