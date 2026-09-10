import numpy as np
import pandas as pd
import multiprocessing as mp
from CalculoIndicesVecino import indices_primeros_vecinos as ipv, indices_segundos_vecinos as isv
from FlipSpin import flip_spin, energia_total_inicial, calcular_Ms
import matplotlib.pyplot as plt

def simular_una_temperatura(S, mcs_max=10000, J1=1, T=1, is_periodica=False):

    J2 = -0.6*J1
    mcs_steps=range(0, mcs_max)

    E0 = energia_total_inicial(S, J1, J2, is_periodica)
    M0 = S.sum()

    suma_E, suma_E2 = 0.0, 0.0
    suma_M, suma_M2 = 0.0, 0.0
    suma_Ms, suma_Ms2, suma_Ms4 = 0.0, 0.0, 0.0

    historial_E = [E0]
    historial_M = [M0]
    historial_Ms = []

    E_prom, E2_prom = [], []
    M_prom, M2_prom = [], []
    Ms_prom, Ms2_prom, Ms4_prom = [], [], []

    E, M = E0, M0

    for mcs in mcs_steps:
        E, M, S = flip_spin(J1, J2, T, S, E, M, is_periodica)
        Ms = calcular_Ms(S)

        historial_E.append(E)
        historial_M.append(M)
        historial_Ms.append(Ms)

        suma_E += E
        suma_E2 += E**2
        suma_M += M
        suma_M2 += M**2
        suma_Ms  += Ms
        suma_Ms2 += Ms**2
        suma_Ms4 += Ms**4
        n = mcs + 1

        E_prom.append( np.abs(suma_E) / n)
        E2_prom.append(suma_E2 / n)
        M_prom.append( np.abs(suma_M) / n)
        M2_prom.append(suma_M2 / n)
        Ms_prom.append(suma_Ms / n)
        Ms2_prom.append(suma_Ms2 / n)
        Ms4_prom.append(suma_Ms4 / n)

    return historial_E, historial_M, historial_Ms, E_prom, E2_prom, M_prom, M2_prom, Ms_prom, Ms2_prom, Ms4_prom, mcs_steps

def graficar_historial(mcs_steps, E_prom, E2_prom, M_prom, M2_prom, Ms_prom, Ms2_prom):

    fig, axs = plt.subplots(6, 1, sharex=True, figsize=(8, 14))

    axs[0].plot(mcs_steps, E_prom)
    axs[0].set_ylabel(r"$\langle E \rangle$")

    axs[1].plot(mcs_steps, E2_prom)
    axs[1].set_ylabel(r"$\langle E^2 \rangle$")

    axs[2].plot(mcs_steps, M_prom)
    axs[2].set_ylabel(r"$\langle M \rangle$")

    axs[3].plot(mcs_steps, M2_prom)
    axs[3].set_ylabel(r"$\langle M^2 \rangle$")

    axs[4].plot(mcs_steps, Ms_prom)
    axs[4].set_ylabel(r"$\langle M_s \rangle$")

    axs[5].plot(mcs_steps, Ms2_prom)
    axs[5].set_ylabel(r"$\langle M_s^2 \rangle$")
    axs[5].set_xlabel("MCS")

    plt.tight_layout()
    plt.show()

def graficar_historial_normalizado(mcs_steps, E_prom, E2_prom, M_prom, M2_prom, Ms_prom, Ms2_prom):
    def normalizar(x):
        x = np.array(x, dtype=float)
        return (x - x.min()) / (x.max() - x.min())

    plt.plot(mcs_steps, normalizar(E_prom), label=r"$\langle E \rangle$")
    plt.plot(mcs_steps, normalizar(E2_prom), label=r"$\langle E^2 \rangle$")
    plt.plot(mcs_steps, normalizar(M_prom), label=r"$\langle M \rangle$")
    plt.plot(mcs_steps, normalizar(M2_prom), label=r"$\langle M^2 \rangle$")
    plt.plot(mcs_steps, normalizar(Ms_prom), label=r"$\langle M_s \rangle$")
    plt.plot(mcs_steps, normalizar(Ms2_prom), label=r"$\langle M_s^2 \rangle$")
    plt.xlabel("MCS")
    plt.ylabel("Valor normalizado")
    plt.legend()
    plt.show()

def calcular_metricas_finales(tao, historial_E, historial_Ms, L, T):
    E_termalizada = historial_E[tao+1:] #Se suma uno porque el arreglo del historial se inicializa con un valor
    E_termalizada_prom = np.sum(E_termalizada) / len(E_termalizada)
    E2_termalizada_prom = np.sum(np.pow(E_termalizada,2)) / len(E_termalizada)

    N = L * L
    Cv = (E2_termalizada_prom - E_termalizada_prom**2) / (N * T**2)

    Ms_termalizado = historial_Ms[tao:]
    Ms_termalizado_prom = np.sum(np.abs(Ms_termalizado)) / len(Ms_termalizado)
    Ms2_termalizado_prom = np.sum(np.pow(Ms_termalizado,2)) / len(Ms_termalizado)
    Ms4_termalizado_prom = np.sum(np.pow(Ms_termalizado, 4)) / len(Ms_termalizado)

    chi_s = (N / T) * (Ms2_termalizado_prom - Ms_termalizado_prom**2)

    U = 1 - Ms4_termalizado_prom / (3 * Ms2_termalizado_prom**2)

    return {
        "T": T, "L": L,
        "E": E_termalizada_prom,
        "Ms": Ms_termalizado_prom,
        "Cv": Cv,
        "chi_s": chi_s,
        "U": U,
    }

def _correr_un_intento(tarea):
    # Corre UNA simulacion completa (un "intento") y devuelve solo sus
    # metricas finales, no el historial completo, para no generar miles
    # de archivos ni de filas innecesarias cuando se promedia sobre muchos
    # intentos independientes.
    l, per, t, mcs_max, tao, J1, intento, semilla = tarea

    # Semilla explicita por tarea: al usar multiprocessing con fork, todos
    # los procesos hijos heredan el mismo estado del generador aleatorio de
    # numpy si no se re-semillan, lo que produciria corridas correlacionadas
    # (no independientes) en vez de muestras estadisticas genuinas.
    np.random.seed(semilla)

    S = np.random.choice([-1, 1], size=(l, l))

    (historial_E, historial_M, historial_Ms,
     E_prom, E2_prom, M_prom, M2_prom,
     Ms_prom, Ms2_prom, Ms4_prom, mcs_steps) = simular_una_temperatura(
        S=S, mcs_max=mcs_max, J1=J1, T=t, is_periodica=per)

    r = calcular_metricas_finales(tao, historial_E, historial_Ms, l, t)
    r["periodica"] = per
    r["intento"] = intento
    return r

def correr_barrido_paralelo(L_lista, T_lista, mcs_max, tao, n_intentos, J1=1, n_procesos=None):
    # Reparte todas las combinaciones (L, frontera, T, intento) entre varios
    # procesos en paralelo, ya que cada intento es completamente independiente
    # de los demas. Devuelve un unico DataFrame con una fila por intento.
    tareas = []
    ss = np.random.SeedSequence()
    n_tareas = len(L_lista) * 2 * len(T_lista) * n_intentos
    semillas = [int(s.generate_state(1)[0]) for s in ss.spawn(n_tareas)]

    idx = 0
    for l in L_lista:
        for per in [False, True]:
            for t in T_lista:
                for intento in range(n_intentos):
                    tareas.append((l, per, t, mcs_max, tao, J1, intento, semillas[idx]))
                    idx += 1

    resultados = []
    with mp.Pool(processes=n_procesos) as pool:
        for i, r in enumerate(pool.imap_unordered(_correr_un_intento, tareas), start=1):
            resultados.append(r)
            if i % 50 == 0 or i == len(tareas):
                print(f"{i}/{len(tareas)} intentos completados", flush=True)

    return pd.DataFrame(resultados)

def _correr_una_cadena(tarea):
    # Corre UNA cadena: recorre todas las temperaturas de mayor a menor para
    # una sola combinacion (L, frontera, intento), heredando la configuracion
    # S de una temperatura como punto de partida de la siguiente (mas fria).
    # Esto ayuda a que el sistema termalice bien a T bajas, en vez de partir
    # siempre de una red aleatoria (recomendacion del profesor).
    l, per, T_lista_desc, mcs_max, tao, J1, intento, semilla = tarea

    np.random.seed(semilla)
    S = np.random.choice([-1, 1], size=(l, l))

    resultados_cadena = []
    for t in T_lista_desc:  # ya debe venir ordenada de mayor a menor
        (historial_E, historial_M, historial_Ms,
         E_prom, E2_prom, M_prom, M2_prom,
         Ms_prom, Ms2_prom, Ms4_prom, mcs_steps) = simular_una_temperatura(
            S=S, mcs_max=mcs_max, J1=J1, T=t, is_periodica=per)
        # S queda modificado "in-place" dentro de simular_una_temperatura
        # (via flip_spin), asi que la siguiente iteracion (T mas fria)
        # continua automaticamente desde donde quedo esta, sin que haga
        # falta reasignarla explicitamente.

        r = calcular_metricas_finales(tao, historial_E, historial_Ms, l, t)
        r["periodica"] = per
        r["intento"] = intento
        resultados_cadena.append(r)

    return resultados_cadena

def correr_barrido_encadenado(L_lista, T_lista, mcs_max, tao, n_intentos, J1=1, n_procesos=None):
    # Version "recocido" (annealing) de correr_barrido_paralelo: en vez de
    # 3200 corridas totalmente independientes, arma n_intentos cadenas
    # independientes por cada (L, frontera), y cada cadena recorre TODAS las
    # temperaturas de mayor a menor en una sola tarea. El paralelismo ahora
    # es entre cadenas, no entre temperaturas individuales.
    T_lista_desc = sorted(T_lista, reverse=True)

    tareas = []
    ss = np.random.SeedSequence()
    n_tareas = len(L_lista) * 2 * n_intentos
    semillas = [int(s.generate_state(1)[0]) for s in ss.spawn(n_tareas)]

    idx = 0
    for l in L_lista:
        for per in [False, True]:
            for intento in range(n_intentos):
                tareas.append((l, per, T_lista_desc, mcs_max, tao, J1, intento, semillas[idx]))
                idx += 1

    total_puntos = len(tareas) * len(T_lista_desc)
    resultados = []
    with mp.Pool(processes=n_procesos) as pool:
        for i, cadena in enumerate(pool.imap_unordered(_correr_una_cadena, tareas), start=1):
            resultados.extend(cadena)
            if i % 20 == 0 or i == len(tareas):
                print(f"{i}/{len(tareas)} cadenas completadas "
                      f"({len(resultados)}/{total_puntos} puntos T)", flush=True)

    return pd.DataFrame(resultados)

def graficar_observable_vs_T(df_libre, df_periodica, columna, ylabel, titulo=None):
    # Grafica una columna (Ms, chi_s, U, E, ...) en funcion de T, con una linea
    # por cada combinacion de tamaño de red (L) y condicion de frontera.
    fig, ax = plt.subplots(figsize=(7, 5))

    for L in sorted(df_libre["L"].unique()):
        sub_libre = df_libre[df_libre["L"] == L].sort_values("T")
        ax.plot(sub_libre["T"], sub_libre[columna], marker="o", label=f"L={L}, libre")

    for L in sorted(df_periodica["L"].unique()):
        sub_per = df_periodica[df_periodica["L"] == L].sort_values("T")
        ax.plot(sub_per["T"], sub_per[columna], marker="s", label=f"L={L}, periodica")

    ax.set_xlabel("T")
    ax.set_ylabel(ylabel)
    if titulo:
        ax.set_title(titulo)
    ax.legend()
    plt.tight_layout()
    plt.show()

def resumir_intentos(df_resultados):
    # Toma el DataFrame con una fila por intento (salida de correr_barrido_paralelo)
    # y devuelve el promedio y el error estandar de la media sobre los N intentos
    # de cada combinacion (L, periodica, T). El error estandar (no la desviacion
    # estandar cruda) es lo que corresponde graficar como barra de error del
    # promedio, ya que decrece con mas intentos (sem = std / sqrt(n)).
    def sem(x):
        return x.std(ddof=1) / np.sqrt(len(x))

    resumen = df_resultados.groupby(["L", "periodica", "T"]).agg(
        E_mean=("E", "mean"), E_sem=("E", sem),
        Ms_mean=("Ms", "mean"), Ms_sem=("Ms", sem),
        Cv_mean=("Cv", "mean"), Cv_sem=("Cv", sem),
        chi_s_mean=("chi_s", "mean"), chi_s_sem=("chi_s", sem),
        U_mean=("U", "mean"), U_sem=("U", sem),
        n_intentos=("Ms", "count"),
    ).reset_index()

    return resumen

def graficar_observable_con_error(resumen, columna, ylabel, titulo=None):
    # Igual que graficar_observable_vs_T, pero con barras de error (sem) y
    # tomando un solo DataFrame "resumen" (salida de resumir_intentos) que
    # ya trae ambas condiciones de frontera en la columna "periodica".
    fig, ax = plt.subplots(figsize=(7, 5))

    for L in sorted(resumen["L"].unique()):
        for periodica, marcador in [(False, "o"), (True, "s")]:
            sub = resumen[(resumen["L"] == L) & (resumen["periodica"] == periodica)].sort_values("T")
            etiqueta = f"L={L}, {'periodica' if periodica else 'libre'}"
            ax.errorbar(sub["T"], sub[f"{columna}_mean"], yerr=sub[f"{columna}_sem"],
                        marker=marcador, capsize=3, label=etiqueta)

    ax.set_xlabel("T")
    ax.set_ylabel(ylabel)
    if titulo:
        ax.set_title(titulo)
    ax.legend()
    plt.tight_layout()
    plt.show()

def graficar_observable_lineas(resumen, columna, ylabel, titulo=None):
    # Version "limpia" de graficar_observable_con_error: sin marcadores de
    # punto ni barras de error, solo la linea. El tamaño de red L se
    # distingue por el tipo de linea (continua/discontinua) y la condicion
    # de frontera por el color.
    fig, ax = plt.subplots(figsize=(7, 5))

    estilos_L = {16: "-", 24: "--"}
    colores_frontera = {False: "tab:blue", True: "tab:red"}

    for L in sorted(resumen["L"].unique()):
        for periodica in [False, True]:
            sub = resumen[(resumen["L"] == L) & (resumen["periodica"] == periodica)].sort_values("T")
            etiqueta = f"L={L}, {'periodica' if periodica else 'libre'}"
            ax.plot(sub["T"], sub[f"{columna}_mean"],
                    linestyle=estilos_L.get(L, "-"),
                    color=colores_frontera[periodica],
                    label=etiqueta)

    ax.set_xlabel("T")
    ax.set_ylabel(ylabel)
    if titulo:
        ax.set_title(titulo)
    ax.legend()
    plt.tight_layout()
    plt.show()

def graficar_observable_paneles(resumen, columna, ylabel, titulo=None, T_min=None, T_max=None):
    # Un panel por condicion de frontera (libre, periodica), cada uno con
    # las lineas de L=16 (azul) y L=24 (roja), sin marcadores. T_min/T_max
    # permiten hacer zoom a una ventana de temperatura, igual que el
    # zoom de U(T) cerca de la transicion.
    colores_L = {16: "tab:blue", 24: "tab:red"}

    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, periodica, nombre in zip(axs, [False, True], ["libre", "periódica"]):
        sub_frontera = resumen[resumen["periodica"] == periodica]
        if T_min is not None:
            sub_frontera = sub_frontera[sub_frontera["T"] >= T_min]
        if T_max is not None:
            sub_frontera = sub_frontera[sub_frontera["T"] <= T_max]

        for L in sorted(sub_frontera["L"].unique()):
            sub = sub_frontera[sub_frontera["L"] == L].sort_values("T")
            ax.plot(sub["T"], sub[f"{columna}_mean"], color=colores_L.get(L), label=f"L={L}")

        ax.set_xlabel("T")
        ax.set_title(f"Frontera {nombre}")
        ax.legend()

    axs[0].set_ylabel(ylabel)
    if titulo:
        fig.suptitle(titulo)
    plt.tight_layout()
    plt.show()

def calcular_metricas_desde_csv(tao, E, Ms, L, T):
    # Version de calcular_metricas_finales para los datos ya guardados en CSV.
    # Ojo: aqui E y Ms SI quedan alineados uno a uno (a diferencia de
    # historial_E/historial_Ms en memoria, el CSV ya no tiene el valor semilla
    # inicial de historial_E), asi que el corte de termalizacion es el mismo
    # indice para ambos arreglos, sin el +1 que usa calcular_metricas_finales.
    E_termalizada = np.asarray(E[tao:])
    Ms_termalizado = np.asarray(Ms[tao:])

    E_termalizada_prom = E_termalizada.mean()
    E2_termalizada_prom = (E_termalizada**2).mean()

    N = L * L
    Cv = (E2_termalizada_prom - E_termalizada_prom**2) / (N * T**2)

    Ms_termalizado_prom = np.abs(Ms_termalizado).mean()
    Ms2_termalizado_prom = (Ms_termalizado**2).mean()
    Ms4_termalizado_prom = (Ms_termalizado**4).mean()

    chi_s = (N / T) * (Ms2_termalizado_prom - Ms_termalizado_prom**2)

    U = 1 - Ms4_termalizado_prom / (3 * Ms2_termalizado_prom**2)

    return {
        "T": T, "L": L,
        "E": E_termalizada_prom,
        "Ms": Ms_termalizado_prom,
        "Cv": Cv,
        "chi_s": chi_s,
        "U": U,
    }
