# Logica de flip_spin revisada con IA
# Logica de energia_total_inicial realizada por la IA al 100%
# Logica de calcular_Ms revisada, corregida y discutida a fondo con IA

import numpy as np
from CalculoIndicesVecino import indices_primeros_vecinos as ipv, indices_segundos_vecinos as isv

def flip_spin(J1, J2, T, S, E, M, is_periodica=False):
    # Simula 1 paso de Monte Carlo (MCS) de un sistema de Ising 2D con
    # interacciones de primer y segundo vecino, con frontera periodica o libre.
    # Para este sistema en cada mcs se recorre toda la red y se intenta hacer un flip de spin en cada sitio
    # y se aceptan siguiendo el algoritmo de Metropolis.

    beta = 1/T # unidades naturales, K_b=1
    L = len(S[0])

    for ix in range(0,L):
        for iy in range(0, L):
            ss = S[ix,iy] # Spin posicion (i, j)
            inn=ipv(ix, iy, L, is_periodica) # Primeros vecinos de la posicion (i,j)
            innn=isv(ix, iy, L, is_periodica) # Segundos vecinos de la posicion (i,j)
            nn = sum(S[i,j] for (i, j) in inn if (i is not None or j is not None)) # Sigma_i para primeros vecinos
            nnn = sum(S[i,j] for (i, j) in innn if (i is not None or j is not None)) # Sigma_i para segundos vecinos
            H = (-J1 * ss * nn) + (-J2 * ss * nnn) # Hamiltoniano original
            ss_new = -ss # se "simula" el flip del spin en la posicion i,j
            H_new = (-J1 * ss_new * nn) + (-J2 * ss_new * nnn) # Nuevo Hamiltoniano despues de hacer el flip del spin
            delta_E = H_new - H # Diferencia de energia antes de hacer el flip y despues de hacer el flip
            #delta_E = 2 * ss * (J1 * nn + J2 * nnn) #¿Equivalente para delta_E?
            W = np.exp(-delta_E * beta) #Factor para aplicar el algoritmo de Metropolis
            if(delta_E <= 0): # si el delta es menor se acepta el flip del spin
                S[ix, iy] = ss_new # se acepta (guarda) el flip en la posicion i,j
                M = M + (ss_new - ss) # Se actualiza el valor de la magnetizacion
                E = E + delta_E # Se actualiza el valor de la energia

            elif(delta_E > 0): # si el delta es mayor se aplica Metropolis
                r = np.random.uniform(0,1) #Se genera un numero aleatorio entre 0 y 1
                if(W>=r): # Si el factor es mayor que el numero aleatorio se acepta el flip del spin
                    S[ix, iy] = ss_new # se acepta (guarda) el flip en la posicion i,j
                    M = M + (ss_new - ss) # Se actualiza el valor de la magnetizacion
                    E = E + delta_E # Se actualiza el valor de la energia
                elif(W<r): # Si el factor es menor se rechaza el flip del spin
                    S[ix, iy] = ss # se deja el valor del spin que se tiene desde el comienzo

    return E, M, S

def flip_spin_whith_h(J1, J2, T, S, E, M, is_periodica=False, h=0.0):
    beta = 1/T # unidades naturales, K_b=1
    L = len(S[0])

    for ix in range(0,L):
        for iy in range(0, L):
            ss = S[ix,iy] # Spin posicion (i, j)
            inn=ipv(ix, iy, L, is_periodica) # Primeros vecinos de la posicion (i,j)
            innn=isv(ix, iy, L, is_periodica) # Segundos vecinos de la posicion (i,j)
            nn = sum(S[i,j] for (i, j) in inn if (i is not None or j is not None)) # Sigma_i para primeros vecinos
            nnn = sum(S[i,j] for (i, j) in innn if (i is not None or j is not None)) # Sigma_i para segundos vecinos
            H = (-J1 * ss * nn) + (-J2 * ss * nnn) + (-h * ss) # Hamiltoniano con campo externo h
            ss_new = -ss # se realiza el flip del spin en la posicion i,j
            H_new = (-J1 * ss_new * nn) + (-J2 * ss_new * nnn) + (-h * ss_new) # Nuevo Hamiltoniano despues de hacer el flip del spin
            delta_E = H_new - H # Diferencia de energia antes de hacer el flip y despues de hacer el flip
            #delta_E = 2 * ss * (J1 * nn + J2 * nnn + h) #¿Equivalente para delta_E?
            W = np.exp(-delta_E * beta) #Factor para aplicar el algoritmo de Metropolis
            if(delta_E <= 0): # si el delta es menor se acepta el flip del spin
                S[ix, iy] = ss_new # se acepta (guarda) el flip en la posicion i,j
                M = M + (ss_new - ss) # Se actualiza el valor de la magnetizacion
                E = E + delta_E # Se actualiza el valor de la energia

            elif(delta_E > 0): # si el delta es mayor se aplica Metropolis
                r = np.random.uniform(0,1) #Se genera un numero aleatorio entre 0 y 1
                if(W>=r): # Si el factor es mayor que el numero aleatorio se acepta el flip del spin
                    S[ix, iy] = ss_new # se acepta (guarda) el flip en la posicion i,j
                    M = M + (ss_new - ss) # Se actualiza el valor de la magnetizacion
                    E = E + delta_E # Se actualiza el valor de la energia
                elif(W<r): # Si el factor es menor se rechaza el flip del spin
                    S[ix, iy] = ss # se deja el valor del spin que se tiene desde el comienzo

    return E, M, S

def energia_total_inicial(S, J1, J2, is_periodica=False):
    L = len(S[0])
    E = 0
    for i in range(L):
        for j in range(L):
            vecinos_nn = ipv(i, j, L, is_periodica)
            vecinos_nnn = isv(i, j, L, is_periodica)

            derecha = vecinos_nn[1]
            abajo = vecinos_nn[2]
            abajo_derecha = vecinos_nnn[1]
            abajo_izquierda = vecinos_nnn[2]

            if derecha[0] is not None:
                E += -J1 * S[i, j] * S[derecha]
            if abajo[0] is not None:
                E += -J1 * S[i, j] * S[abajo]
            if abajo_derecha[0] is not None:
                E += -J2 * S[i, j] * S[abajo_derecha]
            if abajo_izquierda[0] is not None:
                E += -J2 * S[i, j] * S[abajo_izquierda]

    return E

def energia_total_inicial_with_h(S, J1, J2, is_periodica=False, h=0.0):
    # Igual que energia_total_inicial, pero suma tambien el termino de campo
    # externo -h*sum(S_i). A diferencia de los bonds J1/J2 (que conectan dos
    # sitios y hay que tener cuidado de no contarlos dos veces), el termino
    # de campo es de un solo sitio, asi que se suma una sola vez por sitio,
    # sin riesgo de doble conteo.
    E = energia_total_inicial(S, J1, J2, is_periodica)
    E += -h * np.sum(S)
    return E

def calcular_Ms(S):
    # El orden en franjas de este modelo es degenerado: puede aparecer
    # horizontal (por fila) o vertical (por columna), con la misma energia.
    # Si solo midieramos una orientacion, una corrida que ordeno en la otra
    # direccion se veria (incorrectamente) como desordenada (Ms cercano a 0).
    # Por eso se calculan ambas y se toma la mayor.

    L = len(S[0]) # se calcula el tamaño del sistema (LxL)
    N = L * L # numero total de sitios

    # Se crea un arreglo [[0], [1], [2], [3], ..., [L-1]]
    # Cada elemento del arreglo representa el indice de la fila/columna correspondiente.
    indices = np.arange(L).reshape(-1, 1)

    # Se crea un arreglo de 1 y -1. 1 para las filas/columnas pares y -1 para las impares,
    # para poder calcular la magnetizacion absoluta en franjas horizontales y verticales.
    # Por ejemplo, para L=4, el arreglo de indices es [[0], [1], [2], [3]], y el arreglo de signo es [[1], [-1], [1], [-1]].
    # En resumen este arreglo toma la paridad de cada fila/columna y la convierte en un valor de 1 o -1.
    signo = np.where(indices % 2 == 0, 1, -1)

    # las mascaras son la paridad convertida en el arreglo signo
    # para las filas se toma la fila y para las columnas se toma la columna (reshape(1, -1))
    mascara_filas = signo
    mascara_columnas = signo.reshape(1, -1)

    # Se calcula la magnetizacion absoluta en franjas horizontales
    # para eso se multiplica la mascara de filas
    # por la configuracion de spins S (la multiplicacion hace que las filas pares
    # mantengan el signo y las impares cambien de signo). Despues se suma todo para obtener
    # la magnetizacion absoluta en franjas horizontales, luego se divide por 2*N para normalizar y se toma el valor absoluto
    Ms_filas = np.abs(np.sum(mascara_filas * S)) / (2 * N)

    # Se calcula la magnetizacion absoluta en franjas verticales
    # para eso se multiplica la mascara de columnas
    # por la configuracion de spins S (la multiplicacion hace que las columnas pares
    # mantengan el signo y las impares cambien de signo). Despues se suma todo para obtener
    # la magnetizacion absoluta en franjas verticales, luego se divide por 2*N para normalizar y se toma el valor absoluto
    Ms_columnas = np.abs(np.sum(mascara_columnas * S)) / (2 * N)

    # Se toma la mayor de las dos magnetizaciones absolutas en franjas
    return max(Ms_filas, Ms_columnas)

def calcular_Ms_con_h(S):
    # Con campo externo h!=0 la simetria arriba/abajo ya esta rota por el
    # campo, asi que |M_A - M_B| (la definicion de calcular_Ms) deja de ser
    # el parametro de orden correcto: si el campo alinea a las DOS subredes
    # en la misma direccion (a T baja, el campo le gana al acoplamiento
    # J2 antiferromagnetico), esa diferencia se va a cero aunque el sistema
    # este perfectamente ordenado (solo que ferromagneticamente, no en
    # franjas). En su lugar se usa (A + |B|)/2, donde:
    #   - A = la subred con magnetizacion promedio mas alta (la que el
    #     campo alinea de forma confiable, sin ambiguedad de signo)
    #   - B = la otra subred, cuyo signo puede variar segun gane el campo
    #     o el acoplamiento J2 (por eso se usa su valor absoluto, igual que
    #     calcular_Ms usa abs() para no promediar a cero por la eleccion
    #     de dominio)
    # Igual que en calcular_Ms, se prueban franjas horizontales y
    # verticales (orden degenerado) y se toma la mayor.
    L = len(S[0])

    def combinar(m1, m2):
        A, B = max(m1, m2), min(m1, m2)
        return (A + np.abs(B)) / 2

    m_fila_par = S[0::2, :].mean()
    m_fila_impar = S[1::2, :].mean()
    Ms_h_filas = combinar(m_fila_par, m_fila_impar)

    m_col_par = S[:, 0::2].mean()
    m_col_impar = S[:, 1::2].mean()
    Ms_h_columnas = combinar(m_col_par, m_col_impar)

    return max(Ms_h_filas, Ms_h_columnas)
