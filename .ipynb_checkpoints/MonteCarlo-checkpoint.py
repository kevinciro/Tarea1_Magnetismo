import numpy as np
import pandas as pd
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
