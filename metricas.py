# Logica calcular_metricas_finales revisada y corregida con IA
# Logica de resumir_intentos realizada por la IA al 100%

import numpy as np


def calcular_metricas_finales(tao, historial_E, historial_Ms, L, T):
    E_termalizada = historial_E[tao+1:] #Se suma uno porque el arreglo del historial se inicializa con un valor
    E_termalizada_prom = np.sum(E_termalizada) / len(E_termalizada) # calculo de la energia promedio termalizada
    E2_termalizada_prom = np.sum(np.pow(E_termalizada,2)) / len(E_termalizada) # calculo de la energia al cuadrado promedio termalizada

    N = L * L # calculo del numero de sitios en la red
    Cv = (E2_termalizada_prom - E_termalizada_prom**2) / (N * T**2) # calculo del calor especifico

    Ms_termalizado = historial_Ms[tao:] # Se toman los valores de magnetizacion de subred a partir del tiempo de termalizacion
    Ms_termalizado_prom = np.sum(np.abs(Ms_termalizado)) / len(Ms_termalizado) # calculo de la magnetizacion de subred promedio termalizada
    Ms2_termalizado_prom = np.sum(np.pow(Ms_termalizado,2)) / len(Ms_termalizado) # calculo de la magnetizacion de subred al cuadrado promedio termalizada
    Ms4_termalizado_prom = np.sum(np.pow(Ms_termalizado, 4)) / len(Ms_termalizado) # calculo de la magnetizacion de subred a la cuarta potencia promedio termalizada

    chi_s = (N / T) * (Ms2_termalizado_prom - Ms_termalizado_prom**2) # calculo de la susceptibilidad magnetica de subred

    U = 1 - Ms4_termalizado_prom / (3 * Ms2_termalizado_prom**2) # calculo del cumulante de Binder

    return {
        "T": T, "L": L,
        "E": E_termalizada_prom / N, # energia por particula (E/N), no la extensiva
        "Ms": Ms_termalizado_prom,
        "Cv": Cv,
        "chi_s": chi_s,
        "U": U,
    }

def calcular_metricas_finales_with_h(tao, historial_E, historial_M, historial_Ms, L, T):
    E_termalizada = historial_E[tao+1:] #Se suma uno porque el arreglo del historial se inicializa con un valor
    E_termalizada_prom = np.sum(E_termalizada) / len(E_termalizada) # calculo de la energia promedio termalizada
    E2_termalizada_prom = np.sum(np.pow(E_termalizada,2)) / len(E_termalizada) # calculo de la energia al cuadrado promedio termalizada

    N = L * L # calculo del numero de sitios en la red
    Cv = (E2_termalizada_prom - E_termalizada_prom**2) / (N * T**2) # calculo del calor especifico

    M_termalizado = historial_M[tao:] # Se toman los valores de magnetizacion total (extensiva) a partir del tiempo de termalizacion
    M_termalizado_prom = np.sum(np.abs(M_termalizado)) / len(M_termalizado) # calculo de la magnetizacion total promedio termalizada
    M2_termalizado_prom = np.sum(np.pow(M_termalizado,2)) / len(M_termalizado) # calculo de la magnetizacion total al cuadrado promedio termalizada

    chi = (M2_termalizado_prom - M_termalizado_prom**2) / (N * T) # calculo de la susceptibilidad magnetica uniforme (respuesta de M al campo)

    Ms_termalizado = historial_Ms[tao:] # Se toman los valores de magnetizacion de subred a partir del tiempo de termalizacion
    Ms_termalizado_prom = np.sum(np.abs(Ms_termalizado)) / len(Ms_termalizado) # calculo de la magnetizacion de subred promedio termalizada
    Ms2_termalizado_prom = np.sum(np.pow(Ms_termalizado,2)) / len(Ms_termalizado) # calculo de la magnetizacion de subred al cuadrado promedio termalizada
    Ms4_termalizado_prom = np.sum(np.pow(Ms_termalizado, 4)) / len(Ms_termalizado) # calculo de la magnetizacion de subred a la cuarta potencia promedio termalizada

    chi_s = (N / T) * (Ms2_termalizado_prom - Ms_termalizado_prom**2) # calculo de la susceptibilidad magnetica de subred (orden antiferromagnetico)

    U = 1 - Ms4_termalizado_prom / (3 * Ms2_termalizado_prom**2) # calculo del cumulante de Binder de Ms

    return {
        "T": T, "L": L,
        "E": E_termalizada_prom / N, # energia por particula (E/N), no la extensiva
        "M": M_termalizado_prom,
        "Ms": Ms_termalizado_prom,
        "Cv": Cv,
        "chi": chi,
        "chi_s": chi_s,
        "U": U,
    }

def resumir_intentos(df_resultados):
    # Toma el DataFrame con una fila por intento (salida de correr_barrido)
    # y devuelve el promedio y el error estandar de la media sobre los N
    # intentos de cada combinacion (L, periodica, T). El error estandar (no
    # la desviacion estandar cruda) es lo que corresponde graficar como
    # barra de error del promedio, ya que decrece con mas intentos
    # (sem = std / sqrt(n)).
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

def resumir_intentos_with_h(df_resultados):
    # Igual que resumir_intentos, pero para el DataFrame combinado de
    # correr_barrido_with_h: agrupa tambien por h (no solo L, periodica, T),
    # ya que un mismo (L, periodica, T) se visita una vez por cada h del
    # barrido y esas corridas NO deben promediarse juntas. Incluye tanto M
    # (magnetizacion total/extensiva, respuesta al campo) como Ms
    # (magnetizacion de subred, orden antiferromagnetico), con sus
    # respectivas susceptibilidades chi y chi_s, y el cumulante de Binder U.
    def sem(x):
        return x.std(ddof=1) / np.sqrt(len(x))

    resumen = df_resultados.groupby(["L", "periodica", "T", "h"]).agg(
        E_mean=("E", "mean"), E_sem=("E", sem),
        M_mean=("M", "mean"), M_sem=("M", sem),
        Ms_mean=("Ms", "mean"), Ms_sem=("Ms", sem),
        Cv_mean=("Cv", "mean"), Cv_sem=("Cv", sem),
        chi_mean=("chi", "mean"), chi_sem=("chi", sem),
        chi_s_mean=("chi_s", "mean"), chi_s_sem=("chi_s", sem),
        U_mean=("U", "mean"), U_sem=("U", sem),
        n_intentos=("M", "count"),
    ).reset_index()

    return resumen
