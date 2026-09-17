"""
Reanuda main_con_h.py despues de que la corrida original se interrumpiera
(WSL se reinicio a mitad de camino, no fue un error del codigo). Estado en
el que quedo la corrida original:
  - h=0.0: completo, 40/40 intentos guardados en
    ResultadosConH/ResultadosPorCadena/L24/periodica/h_0/Intento1..40/
  - h=0.5: incompleto, solo 8/40 intentos guardados en
    ResultadosConH/ResultadosPorCadena/L24/periodica/h_0.5/Intento1..8/

Este script SOLO corre los 32 intentos que faltan para h=0.5
(Intento9..Intento40), usando el mismo CARPETA_BASE para que cada cadena
quede en su lugar junto a las que ya existen, sin sobreescribir nada
(intento_inicio=8 hace que las nuevas cadenas empiecen en Intento9).

Guarda su propio log (resume_con_h.log, no main_con_h.log) y su propio CSV
parcial (no resultados_main_con_h.csv, que la corrida original nunca llego
a escribir). Cuando esto termine, hay que combinar manualmente los CSV por
cadena de las 40+40 cadenas (h=0 completo + h=0.5 recien completado) para
el analisis final.
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import time
from MonteCarlo import correr_barrido_with_h

L_LISTA = [24]
T_LISTA = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 2.0]
H_LISTA = [0.5]
MCS_MAX = 50_000
INTENTO_INICIO = 8  # ya existen Intento1..Intento8 (indices 0..7)
N_INTENTOS_FALTANTES = 32  # Intento9..Intento40 (indices 8..39)
J1 = 1
CARPETA_BASE = "ResultadosConH"
ARCHIVO_SALIDA = os.path.join(CARPETA_BASE, "resultados_resume_h05.csv")


def obtener_tao(t):
    return 15_000 if t <= 0.5 else 5_000


if __name__ == "__main__":
    n_puntos = N_INTENTOS_FALTANTES * len(T_LISTA)

    print("=" * 70)
    print("RESUME de main_con_h.py: solo faltantes de h=0.5")
    print("=" * 70)
    print(f"L: {L_LISTA}")
    print("Frontera: periodica")
    print(f"h: {H_LISTA}")
    print(f"T ({len(T_LISTA)} valores): {T_LISTA}")
    print(f"mcs_max: {MCS_MAX}")
    print(f"Intentos a correr: {N_INTENTOS_FALTANTES} "
          f"(Intento{INTENTO_INICIO + 1}..Intento{INTENTO_INICIO + N_INTENTOS_FALTANTES})")
    print(f"Total: {N_INTENTOS_FALTANTES} cadenas x {len(T_LISTA)} temperaturas = {n_puntos} puntos")
    print(f"Carpeta de resultados: {CARPETA_BASE}/ (no se toca Intento1..{INTENTO_INICIO}, ya existentes)")
    print("=" * 70, flush=True)

    t0 = time.time()
    df = correr_barrido_with_h(L_LISTA, T_LISTA, MCS_MAX, N_INTENTOS_FALTANTES, tao=obtener_tao,
                                h_lista=H_LISTA, J1=J1, carpeta_base=CARPETA_BASE,
                                intento_inicio=INTENTO_INICIO)
    df.to_csv(ARCHIVO_SALIDA, index=False)
    dt = time.time() - t0

    print("DONE", flush=True)
    print(f"Filas: {df.shape[0]}  Columnas: {df.shape[1]}", flush=True)
    print(f"Tiempo total: {dt/3600:.2f} horas", flush=True)
