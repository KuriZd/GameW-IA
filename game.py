import pygame
import random
import csv
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Constantes del juego
# -----------------------------
WIDTH, HEIGHT = 800, 400
FPS = 30

BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)

POS_INICIAL_X = 50
PLAYER_WIDTH, PLAYER_HEIGHT = 32, 48

BULLET_SIZE = 16
NAVE_SIZE = 64

VEL_JUGADOR = 3
GRAVEDAD = 1
ALTURA_SALTO_INICIAL = 15

# Umbral para mover a la derecha basado en la posición Y de la bala 2
UMBRAL_BALA2_Y = 200

# -----------------------------
# Clase para recopilar datos de entrenamiento
# -----------------------------
class DataCollector:
    def __init__(self):
        # Lista de tuplas: (vel_bala, dist_bala1, bala2_y, dist_bala2, mov_derecha, accion)
        self.data = []

    def add_data(self, vel_bala, dist_bala1, bala2_y, dist_bala2, mov_derecha, accion):
        """
        Agrega una muestra al conjunto de datos.
        """
        self.data.append((vel_bala, dist_bala1, bala2_y, dist_bala2, mov_derecha, accion))

    def has_data(self):
        return len(self.data) > 0

    def save_csv(self, filename="datos_modelo.csv"):
        """
        Guarda todos los datos en un archivo CSV.
        """
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "velocidad_bala",
                "distancia_bala1",
                "bala2_y",
                "distancia_bala2",
                "mov_derecha",
                "accion"
            ])
            writer.writerows(self.data)
        print(f"✅ Datos guardados en {filename}.")

    def train_decision_tree(self, filename="ArbolDecision.joblib", max_depth=5):
        """
        Entrena un árbol de decisión con los datos recopilados (sin usar mov_derecha)
        y lo guarda.
        """
        if not self.has_data():
            print("⚠️ No hay datos para entrenar el modelo de árbol de decisión.")
            return False

        # Construir X solo con las 4 primeras características (sin mov_derecha)
        X = [
            [vel_bala, dist_bala1, bala2_y, dist_bala2]
            for (vel_bala, dist_bala1, bala2_y, dist_bala2, mov_derecha, accion) in self.data
        ]
        y = [accion for (_, _, _, _, _, accion) in self.data]

        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        model.fit(X, y)
        joblib.dump(model, filename)
        print(f"✅ Modelo de Árbol de Decisión entrenado y guardado en {filename}.")
        return True

    def train_neural_network(self, filename="RedNeuronal.joblib", hidden_layer_sizes=(35,)):
        """
        Entrena una Red Neuronal (MLPClassifier) con los datos recopilados
        (sin usar mov_derecha) y la guarda.
        """
        if not self.has_data():
            print("⚠️ No hay datos para entrenar la Red Neuronal.")
            return False

        # Construir X solo con las 4 primeras características (sin mov_derecha)
        X = [
            [vel_bala, dist_bala1, bala2_y, dist_bala2]
            for (vel_bala, dist_bala1, bala2_y, dist_bala2, mov_derecha, accion) in self.data
        ]
        y = [accion for (_, _, _, _, _, accion) in self.data]

        # Convertir a float
        X = [[float(v) for v in row] for row in X]

        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
        model.fit(X, y)
        joblib.dump(model, filename)
        print(f"✅ Red Neuronal entrenada y guardada en {filename}.")
        return True

    def train_knn(self, filename="KNN.joblib", n_neighbors=3):
        """
        Entrena un K-Nearest Neighbors con los datos recopilados
        (sin usar mov_derecha) y lo guarda.
        """
        if not self.has_data():
            print("⚠️ No hay datos para entrenar KNN.")
            return False

        # Construir X solo con las 4 primeras características (sin mov_derecha)
        X = [
            [vel_bala, dist_bala1, bala2_y, dist_bala2]
            for (vel_bala, dist_bala1, bala2_y, dist_bala2, mov_derecha, accion) in self.data
        ]
        y = [accion for (_, _, _, _, _, accion) in self.data]

        # Convertir a float
        X = [[float(v) for v in row] for row in X]

        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X, y)
        joblib.dump(model, filename)
        print(f"✅ KNN entrenado y guardado en {filename}.")
        return True

    def train_linear_regression(self, filename="RegresionLineal.joblib"):
        """
        Entrena un modelo de Regresión (Logistic Regression) con los datos recopilados
        (sin usar mov_derecha) para clasificar acciones y lo guarda.
        """
        if not self.has_data():
            print("⚠️ No hay datos para entrenar Regresión Lineal.")
            return False

        X = [
            [vel_bala, dist_bala1, bala2_y, dist_bala2]
            for (vel_bala, dist_bala1, bala2_y, dist_bala2, mov_derecha, accion) in self.data
        ]
        y = [accion for (_, _, _, _, _, accion) in self.data]

        # Convertir a float
        X = [[float(v) for v in row] for row in X]

        # LogisticRegression usado como clasificador lineal
        model = LogisticRegression(
            multi_class='auto',
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        model.fit(X, y)
        joblib.dump(model, filename)
        print(f"✅ Regresión Lineal (LogisticRegression) entrenada y guardada en {filename}.")
        return True
# -----------------------------
# Clase que representa al jugador
# -----------------------------
class Player:
    def __init__(self, x, y, sprite_frames):
        self.rect = pygame.Rect(x, y, PLAYER_WIDTH, PLAYER_HEIGHT)
        self.frames = sprite_frames
        self.current_frame = 0
        self.frame_speed = 10
        self.frame_count = 0

        self.is_jumping = False
        self.jump_height = ALTURA_SALTO_INICIAL
        self.on_ground = True

    def start_jump(self):
        if self.on_ground:
            self.is_jumping = True
            self.on_ground = False

    def update(self):
        # Animación del sprite
        self.frame_count += 1
        if self.frame_count >= self.frame_speed:
            self.current_frame = (self.current_frame + 1) % len(self.frames)
            self.frame_count = 0

        # Lógica de salto
        if self.is_jumping:
            self.rect.y -= self.jump_height
            self.jump_height -= GRAVEDAD
            if self.rect.y >= HEIGHT - 100:
                self.rect.y = HEIGHT - 100
                self.is_jumping = False
                self.jump_height = ALTURA_SALTO_INICIAL
                self.on_ground = True

    def draw(self, surface):
        surface.blit(self.frames[self.current_frame], (self.rect.x, self.rect.y))


# -----------------------------
# Clase que representa una bala (o proyectil)
# -----------------------------
class Bullet:
    def __init__(self, x, y, dx, dy, image):
        self.rect = pygame.Rect(x, y, BULLET_SIZE, BULLET_SIZE)
        self.vx = dx
        self.vy = dy
        self.image = image
        self.initial_pos = (x, y)

    def reset(self, new_x=None, new_y=None):
        """
        Restaura la posición de la bala a su origen o a valores opcionales.
        """
        if new_x is not None and new_y is not None:
            self.rect.x, self.rect.y = new_x, new_y
        else:
            self.rect.x, self.rect.y = self.initial_pos
        # Si la bala original va en X (vx != 0), volvemos a asignarle velocidad aleatoria
        if self.vx != 0:
            self.vx = random.randint(-8, -3)

    def update(self):
        self.rect.x += self.vx
        self.rect.y += self.vy

    def draw(self, surface):
        surface.blit(self.image, (self.rect.x, self.rect.y))

    def is_off_screen(self):
        return (
            self.rect.x < 0 or self.rect.x > WIDTH or
            self.rect.y < 0 or self.rect.y > HEIGHT
        )


# -----------------------------
# Clase principal que controla el juego
# -----------------------------
class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Juego: Disparo de Bala, Salto, Nave y Menú")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 24)

        # Carga de assets
        self.jugador_frames = [
            pygame.image.load('assets/sprites/mono_frame_1.png').convert_alpha(),
            pygame.image.load('assets/sprites/mono_frame_2.png').convert_alpha(),
            pygame.image.load('assets/sprites/mono_frame_3.png').convert_alpha(),
            pygame.image.load('assets/sprites/mono_frame_4.png').convert_alpha()
        ]
        self.bala_img = pygame.image.load('assets/sprites/purple_ball.png').convert_alpha()
        self.fondo_img = pygame.transform.scale(
            pygame.image.load('assets/game/fondo2.png').convert(), (WIDTH, HEIGHT)
        )
        self.nave_img = pygame.image.load('assets/game/ufo.png').convert_alpha()

        # Instancias de objetos
        self.player = Player(POS_INICIAL_X, HEIGHT - 100, self.jugador_frames)

        # Bala 1: se dispara desde la derecha hacia la izquierda
        self.bullet1 = Bullet(
            WIDTH - 50, HEIGHT - 90,
            dx=random.randint(-8, -3), dy=0, image=self.bala_img
        )
        # Bala 2: se mueve desde arriba hacia abajo
        self.bullet2 = Bullet(50, 0, dx=0, dy=3, image=self.bala_img)

        # "Nave" no dispara, solo se dibuja
        self.nave_rect = pygame.Rect(WIDTH - 100, HEIGHT - 100, NAVE_SIZE, NAVE_SIZE)

        # Fondo "parallax"
        self.fondo_x1 = 0
        self.fondo_x2 = WIDTH

        # Estado del juego
        self.running = True
        self.paused = False
        self.menu_active = True
        self.auto_mode = False
        self.selected_model_name = None
        self.model = None

        # Recolector de datos para entrenamiento
        self.data_collector = DataCollector()

    # ------------------------------------
    # Detección de colisión “mejorada”
    # ------------------------------------
    def detect_collision(self):
        """
        Retorna:
         - 1 si colisiona con bullet1,
         - 2 si colisiona con bullet2,
         - None si no hay colisión.
        """
        if self.player.rect.colliderect(self.bullet1.rect):
            return 1
        if self.player.rect.colliderect(self.bullet2.rect):
            return 2
        return None

    # ------------------------------------
    # Lógica para mostrar mensajes de error
    # ------------------------------------
    def show_error_message(self, mensaje, duration_seconds=2):
        """
        Muestra un mensaje en pantalla por `duration_seconds` segundos o hasta que el usuario presione una tecla.
        """
        self.screen.fill(NEGRO)
        texto = self.font.render(mensaje, True, (255, 50, 50))
        rect_texto = texto.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        self.screen.blit(texto, rect_texto)
        pygame.display.flip()
        start_ticks = pygame.time.get_ticks()
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    waiting = False
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            seconds_passed = (pygame.time.get_ticks() - start_ticks) / 1000
            if seconds_passed >= duration_seconds:
                waiting = False
            self.clock.tick(FPS)

    # ------------------------------------
    # Menú de selección de modelo automático
    # ------------------------------------
    def show_model_menu(self):
        opciones = [
            ("1. Regresión Lineal", "RegresionLineal"),
            ("2. Árboles de Decisión", "ArbolDecision"),
            ("3. Redes Neuronales", "RedNeuronal"),
            ("4. K-Nearest Neighbor", "KNN")
        ]
        selecting = True
        while selecting:
            self.screen.fill(NEGRO)
            y = HEIGHT // 2 - 80
            title_surf = self.font.render("Selecciona el modelo automático:", True, BLANCO)
            self.screen.blit(title_surf, (WIDTH // 4, y - 40))
            for text, _ in opciones:
                surf = self.font.render(text, True, BLANCO)
                self.screen.blit(surf, (WIDTH // 4, y))
                y += 30
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.data_collector.save_csv()
                    pygame.quit()
                    exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        self.selected_model_name = opciones[0][1]
                        selecting = False
                    elif event.key == pygame.K_2:
                        self.selected_model_name = opciones[1][1]
                        selecting = False
                    elif event.key == pygame.K_3:
                        self.selected_model_name = opciones[2][1]
                        selecting = False
                    elif event.key == pygame.K_4:
                        self.selected_model_name = opciones[3][1]
                        selecting = False

        try:
            self.model = joblib.load(f"{self.selected_model_name}.joblib")
            print(f"✅ Modelo cargado: {self.selected_model_name}.joblib")
        except Exception as e:
            print(f"❌ Error al cargar modelo {self.selected_model_name}.joblib: {e}")
            self.show_error_message(
                f"No se pudo cargar {self.selected_model_name}.joblib. Asegúrate de entrenarlo primero.",
                duration_seconds=3
            )
            self.auto_mode = False
            self.model = None

    # ------------------------------------
    # Menú inicial: modo Manual, Automático o Salir
    # ------------------------------------
    def show_main_menu(self):
        self.menu_active = True
        while self.menu_active:
            self.screen.fill(NEGRO)
            texto = self.font.render(
                "Presiona 'A' para Auto, 'M' para Manual, 'R' para Menú, o 'Q' para Salir",
                True, BLANCO
            )
            rect_text = texto.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            self.screen.blit(texto, rect_text)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.data_collector.save_csv()
                    pygame.quit()
                    exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        self.auto_mode = True
                        self.show_model_menu()
                        self.menu_active = False
                    elif event.key == pygame.K_m:
                        self.auto_mode = False
                        self.menu_active = False
                    elif event.key == pygame.K_q:
                        self.data_collector.save_csv()
                        pygame.quit()
                        exit()
                    elif event.key == pygame.K_r:
                        # Si presionan R en el menú, simplemente permanece en el menú
                        self.menu_active = True

    # ------------------------------------
    # Reiniciar el juego tras colisión
    # ------------------------------------
    def restart_game(self):
        self.menu_active = True
        # Reposicionar jugador
        self.player.rect.x, self.player.rect.y = POS_INICIAL_X, HEIGHT - 100
        self.player.is_jumping = False
        self.player.on_ground = True
        self.player.jump_height = ALTURA_SALTO_INICIAL

        # Reposicionar balas
        self.bullet1.reset(new_x=WIDTH - 50, new_y=HEIGHT - 90)
        self.bullet2.reset(new_x=50, new_y=0)

        print("📊 Datos recopilados hasta la colisión:", self.data_collector.data)
        self.show_main_menu()

    # ------------------------------------
    # Guardar datos en cada fotograma (solo en modo manual)
    # ------------------------------------
    def record_data_if_manual(self):
        """
        Registra una muestra en data_collector compuesta por:
        (velocidad_bala1, distancia_bala1, bala2_y, distancia_bala2, mov_derecha, accion)
        """
        keys = pygame.key.get_pressed()
        mov_derecha = 1 if keys[pygame.K_RIGHT] else 0
        salto = 1 if self.player.is_jumping else 0

        dist_bala1 = abs(self.player.rect.x - self.bullet1.rect.x)
        dist_bala2 = abs(self.player.rect.x - self.bullet2.rect.x)
        bala2_y = self.bullet2.rect.y

        if salto and mov_derecha:
            accion = 3
        elif salto:
            accion = 1
        elif mov_derecha:
            accion = 2
        else:
            accion = 0

        self.data_collector.add_data(
            vel_bala=self.bullet1.vx,
            dist_bala1=dist_bala1,
            bala2_y=bala2_y,
            dist_bala2=dist_bala2,
            mov_derecha=mov_derecha,
            accion=accion
        )

    # ------------------------------------
    # Actualización de lógica y dibujo
    # ------------------------------------
    def update_and_draw(self):
        prediccion = None

        # Si estamos en modo automático y hay un modelo cargado
        if self.auto_mode and self.model is not None and not self.paused:
            dist_bala1 = abs(self.player.rect.x - self.bullet1.rect.x)
            dist_bala2 = abs(self.player.rect.x - self.bullet2.rect.x)
            bala2_y = self.bullet2.rect.y

            # 1) Si bala2_y supera el umbral, forzamos mover a la derecha
            if bala2_y > UMBRAL_BALA2_Y:
                prediccion = 2
                print(f"[DEBUG] Forzado mover derecha porque bala2_y={bala2_y} > {UMBRAL_BALA2_Y}")
            else:
                # 2) De lo contrario, dejamos que el modelo decida
                entrada = [[
                    self.bullet1.vx,
                    dist_bala1,
                    bala2_y,
                    dist_bala2
                ]]
                print(
                    f"[DEBUG] Características entrada: "
                    f"vel_bala={self.bullet1.vx}, dist_bala1={dist_bala1}, "
                    f"bala2_y={bala2_y}, dist_bala2={dist_bala2}"
                )
                try:
                    # Convertir entrada a float para Red Neuronal o KNN
                    entrada = [[float(x) for x in entrada[0]]]
                    prediccion = self.model.predict(entrada)[0]
                    print(f"[DEBUG] Predicción del modelo: {prediccion}")
                except Exception as e:
                    print(f"❌ Error al predecir: {e}")
                    prediccion = None

            # 3) Aplicar la predicción
            if prediccion is not None:
                # Si debe saltar
                if prediccion in [1, 3] and self.player.on_ground:
                    print("[DEBUG] Acción: SALTAR")
                    self.player.start_jump()

                # Si debe moverse a la derecha
                if prediccion in [2, 3]:
                    if self.player.rect.x - POS_INICIAL_X < 30:
                        print("[DEBUG] Acción: MOVER DERECHA")
                        self.player.rect.x += VEL_JUGADOR
                    # else:
                    #     print("[DEBUG] Límite de movimiento a la derecha alcanzado")
                else:
                    # Si la predicción no incluye mover a la derecha,
                    # devolvemos al jugador hacia la posición inicial
                    if self.player.rect.x > POS_INICIAL_X:
                        self.player.rect.x -= VEL_JUGADOR
                        if self.player.rect.x < POS_INICIAL_X:
                            self.player.rect.x = POS_INICIAL_X

        # ----------------------------------------------------------
        # 2) Actualizar fondo "parallax"
        # ----------------------------------------------------------
        self.fondo_x1 -= 1
        self.fondo_x2 -= 1
        if self.fondo_x1 <= -WIDTH:
            self.fondo_x1 = WIDTH
        if self.fondo_x2 <= -WIDTH:
            self.fondo_x2 = WIDTH

        # ----------------------------------------------------------
        # 3) Dibujar fondo
        # ----------------------------------------------------------
        self.screen.blit(self.fondo_img, (self.fondo_x1, 0))
        self.screen.blit(self.fondo_img, (self.fondo_x2, 0))

        # ----------------------------------------------------------
        # 4) Actualizar y dibujar jugador
        # ----------------------------------------------------------
        self.player.update()
        self.player.draw(self.screen)

        # ----------------------------------------------------------
        # 5) Dibujar la nave
        # ----------------------------------------------------------
        self.screen.blit(self.nave_img, (self.nave_rect.x, self.nave_rect.y))

        # ----------------------------------------------------------
        # 6) Actualizar y dibujar bala1
        # ----------------------------------------------------------
        self.bullet1.update()
        if self.bullet1.rect.x < 0:  # Si sale por izquierda, reset
            self.bullet1.reset(new_x=WIDTH - 50, new_y=HEIGHT - 90)
        self.bullet1.draw(self.screen)

        # ----------------------------------------------------------
        # 7) Actualizar y dibujar bala2
        # ----------------------------------------------------------
        self.bullet2.update()
        if self.bullet2.rect.y > HEIGHT:  # Si sale por abajo, reset
            self.bullet2.reset(new_x=50, new_y=0)
        self.bullet2.draw(self.screen)

        # ----------------------------------------------------------
        # 8) Detectar colisión (después de mover todo)
        # ----------------------------------------------------------
        collided_bullet = self.detect_collision()
        if collided_bullet is not None:
            print(f"¡Colisión con bala {collided_bullet}!")
            if not self.auto_mode:

                
                # Entrenar todos los modelos: Árbol de Decisión, Red Neuronal y KNN
                dt_ok = self.data_collector.train_decision_tree()
                nn_ok = self.data_collector.train_neural_network()
                knn_ok = self.data_collector.train_knn()
                lr_ok = self.data_collector.train_linear_regression()
                if not (dt_ok or nn_ok or knn_ok or lr_ok):
                    self.show_error_message(
                        "No hay datos para entrenar los modelos.", duration_seconds=2
                    )
                self.restart_game()

        # ----------------------------------------------------------
        # 9) En modo manual, registrar datos cada frame
        # ----------------------------------------------------------
        if (not self.auto_mode) and (not self.paused):
            self.record_data_if_manual()

    # ------------------------------------
    # Bucle principal
    # ------------------------------------
    def run(self):
        self.show_main_menu()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.data_collector.save_csv()
                    self.running = False

                if event.type == pygame.KEYDOWN:
                    if (
                        event.key == pygame.K_UP
                        and self.player.on_ground
                        and not self.paused
                        and not self.auto_mode
                    ):
                        self.player.start_jump()
                    if event.key == pygame.K_p:
                        self.paused = not self.paused
                        print("Juego pausado." if self.paused else "Juego reanudado.")
                    if event.key == pygame.K_q:
                        self.data_collector.save_csv()
                        pygame.quit()
                        exit()
                    if event.key == pygame.K_r:
                        # Al presionar 'R', reiniciamos posiciones y volvemos al menú principal
                        self.player.rect.x, self.player.rect.y = POS_INICIAL_X, HEIGHT - 100
                        self.player.is_jumping = False
                        self.player.on_ground = True
                        self.player.jump_height = ALTURA_SALTO_INICIAL
                        self.bullet1.reset(new_x=WIDTH - 50, new_y=HEIGHT - 90)
                        self.bullet2.reset(new_x=50, new_y=0)
                        self.auto_mode = False
                        self.paused = False
                        self.show_main_menu()

            # Movimiento manual con flecha derecha (solo si no está en auto)
            if not self.paused and not self.auto_mode:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_RIGHT]:
                    if self.player.rect.x - POS_INICIAL_X < 30:
                        self.player.rect.x += VEL_JUGADOR
                else:
                    if self.player.rect.x > POS_INICIAL_X:
                        self.player.rect.x -= VEL_JUGADOR
                        if self.player.rect.x < POS_INICIAL_X:
                            self.player.rect.x = POS_INICIAL_X

            if not self.paused:
                self.update_and_draw()

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()


# ------------------------------------
# Punto de entrada
# ------------------------------------
if __name__ == "__main__":
    game = Game()
    game.run()
