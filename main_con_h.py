
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import time
from MonteCarlo import correr_barrido_with_h
from metricas import resumir_intentos_with_h
from graficas import graficar_observable_h
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Parametros de la corrida (todo lo que se ejecuta queda explicito aqui)
# ---------------------------------------------------------------------------
L_LISTA = [24]
T_LISTA = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 2.0]
# Solo h=0.5: h=0 ya esta completo y correcto (40/40 intentos, con E/N ya
# corregido), no hace falta volver a correrlo. n_intentos=20 (no 40) para
# que esta corrida tome ~6h en vez de ~11h; se puede completar despues con
# los 20 restantes si hace falta mas estadistica.
H_LISTA = [0.5]
MCS_MAX = 50_000
N_INTENTOS = 20
J1 = 1
CARPETA_BASE = "ResultadosConH"
ARCHIVO_SALIDA = os.path.join(CARPETA_BASE, "resultados_h05_20intentos.csv")
ARCHIVO_RESUMEN = os.path.join(CARPETA_BASE, "resumen_h05_20intentos.csv")
CARPETA_GRAFICAS = os.path.join(CARPETA_BASE, "graficas")

def obtener_tao(t):
    # Termalizacion mas larga para T bajas (mas dificil de equilibrar),
    # mas corta para el resto (ya se validó que 5000 es suficiente).
    return 15_000 if t <= 0.5 else 5_000

if __name__ == "__main__":
    n_cadenas = len(L_LISTA) * len(H_LISTA) * N_INTENTOS
    n_puntos = n_cadenas * len(T_LISTA)

    os.makedirs(CARPETA_BASE, exist_ok=True)
    os.makedirs(CARPETA_GRAFICAS, exist_ok=True)

    print("=" * 70)
    print(f"h: {H_LISTA}")
    print("=" * 70)
    print(f"L: {L_LISTA}")
    print("Frontera: periodica")
    print(f"T ({len(T_LISTA)} valores): {T_LISTA}")
    print(f"mcs_max: {MCS_MAX}")
    print(f"n_intentos: {N_INTENTOS}")
    print("tao: 15000 para T<=0.5, 5000 para el resto")
    print(f"Total: {n_cadenas} cadenas x {len(T_LISTA)} temperaturas = {n_puntos} puntos")
    print(f"Carpeta de resultados: {CARPETA_BASE}/ (incluye video + CSV por cada una de las {n_cadenas} cadenas)")
    print("=" * 70, flush=True)

    t0 = time.time()
    df = correr_barrido_with_h(L_LISTA, T_LISTA, MCS_MAX, N_INTENTOS, tao=obtener_tao, h_lista=H_LISTA, J1=J1,
                         carpeta_base=CARPETA_BASE)
    df.to_csv(ARCHIVO_SALIDA, index=False)
    dt = time.time() - t0

    print("Cadenas terminadas, generando resumen y graficas finales...", flush=True)
    resumen = resumir_intentos_with_h(df)
    resumen.to_csv(ARCHIVO_RESUMEN, index=False)

    # Nombres con sufijo "_h05_20intentos" a proposito: esta corrida solo
    # tiene h=0.5 (sin h=0 para comparar), asi que no debe pisar las
    # graficas finales combinadas que ya existen en CARPETA_GRAFICAS.
    for columna, ylabel, nombre_archivo in [
        ("M", r"$M$", "M_T_h05_20intentos.png"),
        ("Ms", r"$M_s$", "Ms_T_h05_20intentos.png"),
        ("chi", r"$\chi$", "chi_T_h05_20intentos.png"),
        ("chi_s", r"$\chi_s$", "chi_s_T_h05_20intentos.png"),
        ("Cv", r"$C_v$", "Cv_T_h05_20intentos.png"),
        ("U", r"$U$", "U_T_h05_20intentos.png"),
        ("E", r"$E$", "E_T_h05_20intentos.png"),
    ]:
        graficar_observable_h(resumen, columna, ylabel, h_lista=H_LISTA, titulo=f"{columna}(T)")
        plt.savefig(os.path.join(CARPETA_GRAFICAS, nombre_archivo), dpi=110)
        plt.close()

    print("DONE", flush=True)
    print(f"Filas: {df.shape[0]}  Columnas: {df.shape[1]}", flush=True)
    print(f"Tiempo total: {dt/3600:.2f} horas", flush=True)
