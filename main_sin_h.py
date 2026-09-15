"""
Barrido de temperaturas SIN campo externo (h=0).

Parametros acordados con el profesor (13-sep-2026), tras revisar que con
100 intentos independientes por (L, frontera, T) el resultado ya era
suficiente, pero hacia falta mas cuidado con la termalizacion a T bajas:

- n_intentos = 40 (mas de lo minimo necesario, para que la distribucion
  muestral de la media se acerque a una normal via CLT, en vez de quedar
  mas cerca de una t de Student como con muestras muy chicas)
- mcs_max = 50_000 (antes 10_000; 5x mas pasos por corrida)
- Rango de T: [0.1, 2.0], mas denso cerca de la transicion (T_N ~ 1.0-1.1
  segun el cruce de U y los picos de Cv/chi_s encontrados antes)
- tao = 15_000 para T <= 0.5 (termalizacion mas larga en frio, donde el
  sistema puede quedar atrapado en paredes de dominio), 5_000 para el resto
- Arranque tibio: cada cadena recorre las temperaturas de mayor a menor,
  heredando la configuracion S de un paso al siguiente (recomendacion del
  profesor), en vez de arrancar cada T desde una red aleatoria nueva.
  Implementado en MonteCarlo.correr_barrido / _correr_una_cadena.

Se corre como script independiente (python3 main_sin_h.py), pensado para
dejarlo en segundo plano.

Estructura de resultados (todo dentro de CARPETA_BASE = "ResultadosSinH"):
  ResultadosSinH/
    resultados_main_sin_h.csv          <- combinado, una fila por (L,frontera,T,intento)
    resumen_main_sin_h.csv             <- promedio +- SEM sobre los intentos, por (L,frontera,T)
    graficas/                          <- Ms(T), chi_s(T), E(T), Cv(T), U(T) por paneles
    ResultadosPorCadena/
      L16/{libre,periodica}/Intento1..40/
        resultados_cadena.csv          <- las 13 filas (una por T) de esa cadena
        evolucion.mp4                  <- video de esa cadena especifica (mismos parametros y semilla)
      L24/{libre,periodica}/Intento1..40/
        ...
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import time
from MonteCarlo import correr_barrido
from metricas import resumir_intentos
from graficas import graficar_observable_paneles
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Parametros de la corrida (todo lo que se ejecuta queda explicito aqui)
# ---------------------------------------------------------------------------
L_LISTA = [16, 24]
T_LISTA = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0]
MCS_MAX = 50_000
N_INTENTOS = 40
J1 = 1
CARPETA_BASE = "ResultadosSinH"
ARCHIVO_SALIDA = os.path.join(CARPETA_BASE, "resultados_main_sin_h.csv")
ARCHIVO_RESUMEN = os.path.join(CARPETA_BASE, "resumen_main_sin_h.csv")
CARPETA_GRAFICAS = os.path.join(CARPETA_BASE, "graficas")


def obtener_tao(t):
    # Termalizacion mas larga para T bajas (mas dificil de equilibrar),
    # mas corta para el resto (ya se validó que 5000 es suficiente).
    return 15_000 if t <= 0.5 else 5_000


if __name__ == "__main__":
    n_cadenas = len(L_LISTA) * 2 * N_INTENTOS
    n_puntos = n_cadenas * len(T_LISTA)

    os.makedirs(CARPETA_BASE, exist_ok=True)
    os.makedirs(CARPETA_GRAFICAS, exist_ok=True)

    print("=" * 70)
    print("BARRIDO SIN CAMPO EXTERNO (h=0)")
    print("=" * 70)
    print(f"L: {L_LISTA}")
    print("Fronteras: libre, periodica")
    print(f"T ({len(T_LISTA)} valores): {T_LISTA}")
    print(f"mcs_max: {MCS_MAX}")
    print(f"n_intentos: {N_INTENTOS}")
    print("tao: 15000 para T<=0.5, 5000 para el resto")
    print(f"Total: {n_cadenas} cadenas x {len(T_LISTA)} temperaturas = {n_puntos} puntos")
    print(f"Carpeta de resultados: {CARPETA_BASE}/ (incluye video + CSV por cada una de las {n_cadenas} cadenas)")
    print("=" * 70, flush=True)

    t0 = time.time()
    df = correr_barrido(L_LISTA, T_LISTA, MCS_MAX, N_INTENTOS, tao=obtener_tao, J1=J1,
                         carpeta_base=CARPETA_BASE)
    df.to_csv(ARCHIVO_SALIDA, index=False)
    dt = time.time() - t0

    print("Cadenas terminadas, generando resumen y graficas finales...", flush=True)
    resumen = resumir_intentos(df)
    resumen.to_csv(ARCHIVO_RESUMEN, index=False)

    for columna, ylabel, nombre_archivo in [
        ("Ms", r"$M_s$", "Ms_T_paneles.png"),
        ("chi_s", r"$\chi_s$", "chi_s_T_paneles.png"),
        ("Cv", r"$C_v$", "Cv_T_paneles.png"),
        ("U", r"$U$", "U_T_paneles.png"),
        ("E", r"$E$", "E_T_paneles.png"),
    ]:
        graficar_observable_paneles(resumen, columna, ylabel, titulo=f"{columna}(T)")
        plt.savefig(os.path.join(CARPETA_GRAFICAS, nombre_archivo), dpi=110)
        plt.close()

    print("DONE", flush=True)
    print(f"Filas: {df.shape[0]}  Columnas: {df.shape[1]}", flush=True)
    print(f"Tiempo total: {dt/3600:.2f} horas", flush=True)
