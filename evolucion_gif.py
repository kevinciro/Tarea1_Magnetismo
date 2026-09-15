"""
Genera un video mostrando como evoluciona la configuracion de la red
(spines) a medida que se enfria el sistema, temperatura por temperatura,
para que el profesor pueda ver la transicion de orden -> desorden de forma
visual. Se guarda como .mp4 (no .gif) para poder pausar y avanzar cuadro a
cuadro con cualquier reproductor de video.

Es una sola cadena (una sola corrida), no un barrido estadistico: el
objetivo es ilustrativo, no obtener promedios. Por eso mcs_max es mas chico
que en main_sin_h.py (bastante mas rapido de correr).

Cada sitio de la red se dibuja como el perimetro de una celda (sin relleno
de color) con una flecha adentro: apunta hacia arriba si el spin es +1
(roja) y hacia abajo si es -1 (azul). Como la red se enfria de mayor a
menor T (arranque tibio, igual que en el resto del proyecto), el video debe
mostrar flechas mezcladas (arriba/abajo, rojo/azul) al principio, y
alineadas en franjas al final, si el sistema efectivamente ordena.

En el titulo de cada cuadro se muestra tambien Ms (magnetizacion de subred)
calculada en ese momento, para relacionar el patron visual con el numero.

Nota tecnica: ffmpeg no esta instalado en el sistema (matplotlib.animation
lo necesitaria para exportar video), asi que el video se arma directamente
con OpenCV (cv2), escribiendo cada cuadro ya renderizado por matplotlib.
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no necesita pantalla, solo guarda el archivo

from MonteCarlo import simular_una_temperatura
from FlipSpin import calcular_Ms
from graficas import generar_video

# ---------------------------------------------------------------------------
# Parametros
# ---------------------------------------------------------------------------
L = 24
IS_PERIODICA = True
J1 = 1
T_LISTA = [2.0, 1.7, 1.5, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1]  # de mayor a menor
MCS_MAX = 20_000  # mas chico que en main_sin_h.py: aqui solo es ilustrativo
SEMILLA = 42
ARCHIVO_SALIDA = "evolucion_L24_periodica.mp4"
FPS = 1  # cada temperatura (cuadro) dura 1 segundo en el video


def obtener_tao(t):
    # Nota: esta funcion no se usa actualmente en capturar_evolucion (se
    # toma la configuracion final de S directamente, sin recorte de
    # termalizacion). Se deja aqui con un valor acorde a MCS_MAX por si se
    # llega a usar mas adelante.
    return 4_000 if t <= 0.5 else 1_500


def capturar_evolucion():
    np.random.seed(SEMILLA)
    S = np.random.choice([-1, 1], size=(L, L))

    cuadros = []  # lista de (T, copia de S, Ms)
    for t in T_LISTA:
        simular_una_temperatura(S=S, mcs_max=MCS_MAX, J1=J1, T=t, is_periodica=IS_PERIODICA)
        # S queda modificado "in-place"; se guarda una copia porque S se
        # sigue modificando en las siguientes temperaturas.
        Ms_actual = calcular_Ms(S)
        cuadros.append((t, S.copy(), Ms_actual))
        print(f"T={t:.2f}  Ms={Ms_actual:.4f}", flush=True)

    return cuadros


if __name__ == "__main__":
    print(f"L={L}, periodica={IS_PERIODICA}, mcs_max={MCS_MAX}, {len(T_LISTA)} temperaturas")
    t0 = time.time()
    cuadros = capturar_evolucion()
    generar_video(cuadros, ARCHIVO_SALIDA, fps=FPS)
    dt = time.time() - t0
    print(f"Video guardado en {ARCHIVO_SALIDA} ({dt:.1f} s)")
