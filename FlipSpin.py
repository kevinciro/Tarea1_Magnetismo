import numpy as np
from CalculoIndicesVecino import indices_primeros_vecinos as ipv, indices_segundos_vecinos as isv

def flip_spin(J1, J2, T, S, E, M, is_periodica=False):
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
            H = (-J1 * ss * nn) + (-J2 * ss * nnn) + (-h * ss) # Hamiltoniano original
            ss_new = -ss # se "simula" el flip del spin en la posicion i,j
            H_new = (-J1 * ss_new * nn) + (-J2 * ss_new * nnn) + (-h * ss_new) # Nuevo Hamiltoniano despues de hacer el flip del spin
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
    L = len(S[0])
    N = L * L
    indices = np.arange(L).reshape(-1, 1)
    signo = np.where(indices % 2 == 0, 1, -1)

    mascara_filas = signo            # franjas horizontales (alternan por fila)
    mascara_columnas = signo.reshape(1, -1)  # franjas verticales (alternan por columna)

    Ms_filas = np.abs(np.sum(mascara_filas * S)) / (2 * N)
    Ms_columnas = np.abs(np.sum(mascara_columnas * S)) / (2 * N)

    return max(Ms_filas, Ms_columnas)
