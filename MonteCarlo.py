# Logica de simular_una_temperatura y _correr_una_cadena revisada y corregida con IA
# Logica de correr_barrido realizada por la IA al 100%

import os
import numpy as np
import pandas as pd
import multiprocessing as mp
from FlipSpin import flip_spin, energia_total_inicial, calcular_Ms, flip_spin_whith_h, energia_total_inicial_with_h, calcular_Ms_con_h
from metricas import calcular_metricas_finales, calcular_metricas_finales_with_h
from graficas import generar_video


def simular_una_temperatura(S, mcs_max=10000, J1=1, T=1, is_periodica=False):
    # Simula todos los mcs para una temperatura
    # interacciones de primer y segundo vecino, con frontera periodica o libre.
    # En cada paso se intenta hacer el flip de spin en cada sitio de la red
    # y se aceptan siguiendo el algoritmo de Metropolis.

    J2 = -0.6*J1 # Se define J2 en terminos del J1
    mcs_steps=range(0, mcs_max) # Arreglo para guardar el tiempo (mcs)

    E0 = energia_total_inicial(S, J1, J2, is_periodica) # Energia inicial del sistema
    M0 = S.sum() # Magnetizacion inicial del sistema

    suma_E, suma_E2 = 0.0, 0.0 # variables para guardar la suma de energia y energia al cuadrado
    suma_M, suma_M2 = 0.0, 0.0 # variables para guardar la suma de magnetizacion y magnetizacion al cuadrado
    suma_Ms, suma_Ms2, suma_Ms4 = 0.0, 0.0, 0.0 # variables para guardar la suma de magnetizacion absoluta, magnetizacion absoluta al cuadrado y magnetizacion absoluta a la cuarta potencia

    historial_E = [E0] # lista para guardar el historial de energia
    historial_M = [M0]  # lista para guardar el historial de magnetizacion
    historial_Ms = [] # lista para guardar el historial de magnetizacion absoluta

    E_prom, E2_prom = [], [] # listas para guardar el promedio de energia y energia al cuadrado
    M_prom, M2_prom = [], [] # listas para guardar el promedio de magnetizacion y magnetizacion al cuadrado
    Ms_prom, Ms2_prom, Ms4_prom = [], [], [] # listas para guardar el promedio de magnetizacion absoluta, magnetizacion absoluta al cuadrado y magnetizacion absoluta a la cuarta potencia

    E, M = E0, M0 # Se inicializan las variables de energia y magnetizacion con los valores iniciales

    for mcs in mcs_steps: # para cada paso de Monte Carlo
        E, M, S = flip_spin(J1, J2, T, S, E, M, is_periodica) # Se realiza el flip de spin y se actualizan los valores de energia, magnetizacion y la configuracion de spins
        Ms = calcular_Ms(S) # Se calcula la magnetizacion absoluta

        historial_E.append(E) # Se guarda el valor de energia en el historial
        historial_M.append(M) # Se guarda el valor de magnetizacion en el historial
        historial_Ms.append(Ms) # Se guarda el valor de magnetizacion absoluta en el historial

        suma_E += E # Se actualiza la suma de energia
        suma_E2 += E**2 # Se actualiza la suma de energia al cuadrado
        suma_M += M # Se actualiza la suma de magnetizacion
        suma_M2 += M**2 # Se actualiza la suma de magnetizacion al cuadrado
        suma_Ms  += Ms # Se actualiza la suma de magnetizacion absoluta
        suma_Ms2 += Ms**2 # Se actualiza la suma de magnetizacion absoluta al cuadrado
        suma_Ms4 += Ms**4 # Se actualiza la suma de magnetizacion absoluta a la cuarta potencia
        n = mcs + 1 # Se actualiza el numero de pasos de Monte Carlo realizados (mcs + 1 porque se empieza en 0)

        E_prom.append( np.abs(suma_E) / n) # Se guarda el promedio de energia
        E2_prom.append(suma_E2 / n) # Se guarda el promedio de energia al cuadrado
        M_prom.append( np.abs(suma_M) / n) # Se guarda el promedio de magnetizacion
        M2_prom.append(suma_M2 / n) # Se guarda el promedio de magnetizacion al cuadrado
        Ms_prom.append(suma_Ms / n) # Se guarda el promedio de magnetizacion absoluta
        Ms2_prom.append(suma_Ms2 / n) # Se guarda el promedio de magnetizacion absoluta al cuadrado
        Ms4_prom.append(suma_Ms4 / n) # Se guarda el promedio de magnetizacion absoluta a la cuarta potencia

    # Se devuelve el historial de energia,
    # magnetizacion y magnetizacion absoluta,
    # asi como los promedios de energia, energia al cuadrado,
    # magnetizacion, magnetizacion al cuadrado, magnetizacion absoluta,
    # magnetizacion absoluta al cuadrado y magnetizacion absoluta a la cuarta potencia,
    # junto con el arreglo de pasos de Monte Carlo
    return historial_E, historial_M, historial_Ms, E_prom, E2_prom, M_prom, M2_prom, Ms_prom, Ms2_prom, Ms4_prom, mcs_steps

def simular_una_temperatura_with_h(S, mcs_max=10000, J1=1, T=1, h=0.0, is_periodica=False):
    # Simula todos los mcs para una temperatura
    # interacciones de primer y segundo vecino, con frontera periodica o libre.
    # En cada paso se intenta hacer el flip de spin en cada sitio de la red
    # y se aceptan siguiendo el algoritmo de Metropolis.

    J2 = -0.6*J1 # Se define J2 en terminos del J1
    mcs_steps=range(0, mcs_max) # Arreglo para guardar el tiempo (mcs)

    E0 = energia_total_inicial_with_h(S, J1, J2, is_periodica, h) # Energia inicial del sistema
    M0 = S.sum() # Magnetizacion inicial del sistema

    suma_E, suma_E2 = 0.0, 0.0 # variables para guardar la suma de energia y energia al cuadrado
    suma_M, suma_M2 = 0.0, 0.0 # variables para guardar la suma de magnetizacion y magnetizacion al cuadrado
    suma_Ms, suma_Ms2, suma_Ms4 = 0.0, 0.0, 0.0 # variables para guardar la suma de magnetizacion de subred, al cuadrado y a la cuarta potencia

    historial_E = [E0] # lista para guardar el historial de energia
    historial_M = [M0]  # lista para guardar el historial de magnetizacion
    historial_Ms = [] # lista para guardar el historial de magnetizacion de subred

    E_prom, E2_prom = [], [] # listas para guardar el promedio de energia y energia al cuadrado
    M_prom, M2_prom = [], [] # listas para guardar el promedio de magnetizacion y magnetizacion al cuadrado
    Ms_prom, Ms2_prom, Ms4_prom = [], [], [] # listas para guardar el promedio de magnetizacion de subred, al cuadrado y a la cuarta potencia
    E, M = E0, M0 # Se inicializan las variables de energia y magnetizacion con los valores iniciales

    for mcs in mcs_steps: # para cada paso de Monte Carlo
        E, M, S = flip_spin_whith_h(J1, J2, T, S, E, M, is_periodica, h) # Se realiza el flip de spin y se actualizan los valores de energia, magnetizacion y la configuracion de spins
        Ms = calcular_Ms_con_h(S) # Se calcula la magnetizacion de subred con la definicion (A + |B|)/2, correcta cuando el campo rompe la simetria arriba/abajo

        historial_E.append(E) # Se guarda el valor de energia en el historial
        historial_M.append(M) # Se guarda el valor de magnetizacion en el historial
        historial_Ms.append(Ms) # Se guarda el valor de magnetizacion de subred en el historial

        suma_E += E # Se actualiza la suma de energia
        suma_E2 += E**2 # Se actualiza la suma de energia al cuadrado
        suma_M += M # Se actualiza la suma de magnetizacion
        suma_M2 += M**2 # Se actualiza la suma de magnetizacion al cuadrado
        suma_Ms += Ms # Se actualiza la suma de magnetizacion de subred
        suma_Ms2 += Ms**2 # Se actualiza la suma de magnetizacion de subred al cuadrado
        suma_Ms4 += Ms**4 # Se actualiza la suma de magnetizacion de subred a la cuarta potencia
        n = mcs + 1 # Se actualiza el numero de pasos de Monte Carlo realizados (mcs + 1 porque se empieza en 0)

        E_prom.append( np.abs(suma_E) / n) # Se guarda el promedio de energia
        E2_prom.append(suma_E2 / n) # Se guarda el promedio de energia al cuadrado
        M_prom.append( np.abs(suma_M) / n) # Se guarda el promedio de magnetizacion
        M2_prom.append(suma_M2 / n) # Se guarda el promedio de magnetizacion al cuadrado
        Ms_prom.append(suma_Ms / n) # Se guarda el promedio de magnetizacion de subred
        Ms2_prom.append(suma_Ms2 / n) # Se guarda el promedio de magnetizacion de subred al cuadrado
        Ms4_prom.append(suma_Ms4 / n) # Se guarda el promedio de magnetizacion de subred a la cuarta potencia

    # Se devuelve el historial de energia, magnetizacion y magnetizacion de
    # subred, asi como los promedios de energia, energia al cuadrado,
    # magnetizacion, magnetizacion al cuadrado, magnetizacion de subred,
    # magnetizacion de subred al cuadrado y a la cuarta potencia,
    # junto con el arreglo de pasos de Monte Carlo
    return historial_E, historial_M, historial_Ms, E_prom, E2_prom, M_prom, M2_prom, Ms_prom, Ms2_prom, Ms4_prom, mcs_steps

def _ruta_cadena(carpeta_base, l, per, intento, h=None):
    # ResultadosPorCadena/L{l}/{libre|periodica}/Intento{n} (1-indexado, mas
    # amigable de leer que el "intento" interno que arranca en 0).
    # Si h no es None (barrido con campo externo), se agrega un nivel
    # h_{h}/ antes de Intento{n}: sin esto, dos barridos con distinto h pero
    # el mismo numero de intento pisarian la misma carpeta (mismo CSV y
    # video sobrescritos entre si).
    frontera_str = "periodica" if per else "libre"
    partes = [carpeta_base, "ResultadosPorCadena", f"L{l}", frontera_str]
    if h is not None:
        partes.append(f"h_{h:g}")
    partes.append(f"Intento{intento + 1}")
    carpeta = os.path.join(*partes)
    os.makedirs(carpeta, exist_ok=True)
    return carpeta

def _correr_una_cadena(tarea):
    # Corre una cadena de Markov completa (todos los mcs para cada temperaturas):
    # recorre T_lista_desc (de mayor a menor) para
    # una sola combinacion (L, frontera, intento), heredando la configuracion
    # S de una temperatura como punto de partida de la siguiente (la cadena de Markov se forma de modo que
    # la configuracion S de la temperatura anterior es la configuracion inicial de la siguiente temperatura).
    # Devuelve una lista de dicts, uno por temperatura visitada.
    #
    # Si carpeta_base no es None, ademas guarda por su cuenta (dentro de
    # este mismo proceso trabajador) el CSV de esta cadena y un video de su
    # evolucion, en ResultadosPorCadena/L{l}/{frontera}/Intento{n}/.
    l, per, T_lista_desc, mcs_max, J1, intento, semilla, tao_regla, carpeta_base = tarea

    # Semilla explicita por tarea: al usar multiprocessing con fork, todos
    # los procesos hijos heredan el mismo estado del generador aleatorio de
    # numpy si no se re-semillan, lo que produciria corridas correlacionadas
    # (no independientes) en vez de muestras estadisticas genuinas.
    np.random.seed(semilla)
    S = np.random.choice([-1, 1], size=(l, l))

    guardar_por_cadena = carpeta_base is not None
    resultados_cadena = []
    cuadros_video = [] if guardar_por_cadena else None  # (T, copia de S, Ms instantaneo)

    for t in T_lista_desc:  # debe venir ordenada de mayor a menor
        (historial_E, historial_M, historial_Ms,
         E_prom, E2_prom, M_prom, M2_prom,
         Ms_prom, Ms2_prom, Ms4_prom, mcs_steps) = simular_una_temperatura(
            S=S, mcs_max=mcs_max, J1=J1, T=t, is_periodica=per)
        # S queda modificado "in-place" dentro de simular_una_temperatura
        # (via flip_spin), asi que la siguiente iteracion (T mas fria)
        # continua automaticamente desde donde quedo esta.

        tao = tao_regla(t) if callable(tao_regla) else tao_regla
        r = calcular_metricas_finales(tao, historial_E, historial_Ms, l, t)
        r["periodica"] = per
        r["intento"] = intento
        r["tao"] = tao
        resultados_cadena.append(r)

        if guardar_por_cadena:
            # Ms instantaneo de ESTA configuracion puntual (para el video),
            # distinto de r["Ms"] (promedio termalizado sobre muchas
            # muestras, usado para las estadisticas del barrido).
            cuadros_video.append((t, S.copy(), calcular_Ms(S)))

    if guardar_por_cadena:
        carpeta = _ruta_cadena(carpeta_base, l, per, intento)
        pd.DataFrame(resultados_cadena).to_csv(
            os.path.join(carpeta, "resultados_cadena.csv"), index=False)
        generar_video(cuadros_video, os.path.join(carpeta, "evolucion.mp4"), fps=1)

    return resultados_cadena

def _correr_una_cadena_with_h(tarea):
    # Corre una cadena de Markov completa (todos los mcs para cada temperaturas):
    # recorre T_lista_desc (de mayor a menor) para
    # una sola combinacion (L, frontera, intento), heredando la configuracion
    # S de una temperatura como punto de partida de la siguiente (la cadena de Markov se forma de modo que
    # la configuracion S de la temperatura anterior es la configuracion inicial de la siguiente temperatura).
    # Devuelve una lista de dicts, uno por temperatura visitada.
    #
    # Si carpeta_base no es None, ademas guarda por su cuenta (dentro de
    # este mismo proceso trabajador) el CSV de esta cadena y un video de su
    # evolucion, en ResultadosPorCadena/L{l}/{frontera}/Intento{n}/.
    l, per, T_lista_desc, mcs_max, J1, intento, semilla, tao_regla, h, carpeta_base = tarea

    # Semilla explicita por tarea: al usar multiprocessing con fork, todos
    # los procesos hijos heredan el mismo estado del generador aleatorio de
    # numpy si no se re-semillan, lo que produciria corridas correlacionadas
    # (no independientes) en vez de muestras estadisticas genuinas.
    np.random.seed(semilla)
    S = np.random.choice([-1, 1], size=(l, l))

    guardar_por_cadena = carpeta_base is not None
    resultados_cadena = []
    cuadros_video = [] if guardar_por_cadena else None  # (T, copia de S, Ms instantaneo)

    for t in T_lista_desc:  # debe venir ordenada de mayor a menor
        (historial_E, historial_M, historial_Ms,
         E_prom, E2_prom, M_prom, M2_prom,
         Ms_prom, Ms2_prom, Ms4_prom, mcs_steps) = simular_una_temperatura_with_h(
            S=S, mcs_max=mcs_max, J1=J1, T=t, h=h, is_periodica=per)
        # S queda modificado "in-place" dentro de simular_una_temperatura
        # (via flip_spin), asi que la siguiente iteracion (T mas fria)
        # continua automaticamente desde donde quedo esta.

        tao = tao_regla(t) if callable(tao_regla) else tao_regla
        r = calcular_metricas_finales_with_h(tao, historial_E, historial_M, historial_Ms, l, t)
        r["periodica"] = per
        r["intento"] = intento
        r["tao"] = tao
        r["h"] = h
        resultados_cadena.append(r)

        if guardar_por_cadena:
            # M instantaneo (normalizado por sitio, S.sum()/N) de ESTA
            # configuracion puntual (para el video), distinto de r["M"]
            # (promedio termalizado sobre muchas muestras, usado para las
            # estadisticas del barrido). Con campo externo se muestra M
            # (magnetizacion total) en vez de Ms, ya que es la cantidad que
            # el campo satura visiblemente al bajar T.
            cuadros_video.append((t, S.copy(), S.sum() / (l * l)))

    if guardar_por_cadena:
        carpeta = _ruta_cadena(carpeta_base, l, per, intento, h=h)
        pd.DataFrame(resultados_cadena).to_csv(
            os.path.join(carpeta, "resultados_cadena.csv"), index=False)
        generar_video(cuadros_video, os.path.join(carpeta, "evolucion.mp4"), fps=1,
                      nombre_valor="M", ylabel_valor=r"$M$", ylim_valor=(-1.05, 1.05))

    return resultados_cadena

def correr_barrido(L_lista, T_lista, mcs_max, n_intentos, tao, J1=1, n_procesos=None, carpeta_base=None):
    # Ejecuta una cadena de Markov completa en cada nucleo de CPU disponible
    # armando un numero de n_intentos de cadenas independientes por cada (L, frontera), y cada
    # cadena recorre TODAS las temperaturas de mayor a menor en una sola
    # tarea (arranque tibio, tal como se definio en _correr_una_cadena).
    # En resumen el paralelismo es entre cadenas recorriendo los parametros y el arreglo de temperaturas,
    # cada cadena se ejecuta en un nucleo llamando a _correr_una_cadena.
    #
    # carpeta_base: si no es None, cada cadena guarda su propio CSV y video
    # dentro de carpeta_base/ResultadosPorCadena/... (ver _correr_una_cadena).
    # Si es None (por defecto), no se guarda nada por cadena, solo se
    # devuelve el DataFrame combinado, igual que antes.

    T_lista_desc = sorted(T_lista, reverse=True)

    tareas = []
    ss = np.random.SeedSequence()
    n_tareas = len(L_lista) * 2 * n_intentos
    semillas = [int(s.generate_state(1)[0]) for s in ss.spawn(n_tareas)]

    idx = 0
    for l in L_lista:
        for per in [False, True]:
            for intento in range(n_intentos):
                tareas.append((l, per, T_lista_desc, mcs_max, J1, intento, semillas[idx], tao, carpeta_base))
                idx += 1

    total_puntos = len(tareas) * len(T_lista_desc)
    resultados = []
    with mp.Pool(processes=n_procesos) as pool:
        for i, cadena in enumerate(pool.imap_unordered(_correr_una_cadena, tareas), start=1):
            resultados.extend(cadena)
            if i % 5 == 0 or i == len(tareas):
                print(f"{i}/{len(tareas)} cadenas completadas "
                      f"({len(resultados)}/{total_puntos} puntos T)", flush=True)

    return pd.DataFrame(resultados)

def correr_barrido_with_h(L_lista, T_lista, mcs_max, n_intentos, tao, h_lista, J1=1, n_procesos=None, carpeta_base=None, intento_inicio=0):
    # Ejecuta una cadena de Markov completa en cada nucleo de CPU disponible
    # armando un numero de n_intentos de cadenas independientes por cada (L, frontera), y cada
    # cadena recorre TODAS las temperaturas de mayor a menor en una sola
    # tarea (arranque tibio, tal como se definio en _correr_una_cadena).
    # En resumen el paralelismo es entre cadenas recorriendo los parametros y el arreglo de temperaturas,
    # cada cadena se ejecuta en un nucleo llamando a _correr_una_cadena.
    #
    # carpeta_base: si no es None, cada cadena guarda su propio CSV y video
    # dentro de carpeta_base/ResultadosPorCadena/... (ver _correr_una_cadena).
    # Si es None (por defecto), no se guarda nada por cadena, solo se
    # devuelve el DataFrame combinado, igual que antes.
    #
    # intento_inicio: indice (0-based) del primer intento a correr. Permite
    # reanudar un barrido interrumpido corriendo solo los intentos que
    # faltan (n_intentos = cuantos faltan, no el total), sin pisar las
    # carpetas Intento1..Intento{intento_inicio} que ya existen de una
    # corrida anterior.

    T_lista_desc = sorted(T_lista, reverse=True)

    tareas = []
    ss = np.random.SeedSequence()
    n_tareas = len(L_lista) * len(h_lista) * n_intentos
    semillas = [int(s.generate_state(1)[0]) for s in ss.spawn(n_tareas)]

    idx = 0
    for l in L_lista:
        for h in h_lista:
            for intento in range(intento_inicio, intento_inicio + n_intentos):
                tareas.append((l, True, T_lista_desc, mcs_max, J1, intento, semillas[idx], tao, h, carpeta_base))
                idx += 1

    total_puntos = len(tareas) * len(T_lista_desc)
    resultados = []
    with mp.Pool(processes=n_procesos) as pool:
        for i, cadena in enumerate(pool.imap_unordered(_correr_una_cadena_with_h, tareas), start=1):
            resultados.extend(cadena)
            if i % 5 == 0 or i == len(tareas):
                print(f"{i}/{len(tareas)} cadenas completadas "
                      f"({len(resultados)}/{total_puntos} puntos T)", flush=True)

    return pd.DataFrame(resultados)
