# Logica corregida con IA

def _fila_columna_vecinas(i, j, L, is_periodica):
    # Calcula, de forma independiente, a qué fila corresponde moverse
    # "arriba" y "abajo", y a qué columna corresponde moverse
    # "izquierda" y "derecha". Si no es periodica y no existe el vecino
    # en esa direccion, devuelve None para ese valor.
    if(is_periodica):
        fila_arriba = L-1 if i==0 else i-1
        fila_abajo = 0 if i==L-1 else i+1
        col_izq = L-1 if j==0 else j-1
        col_der = 0 if j==L-1 else j+1
    else:
        fila_arriba = None if i==0 else i-1
        fila_abajo = None if i==L-1 else i+1
        col_izq = None if j==0 else j-1
        col_der = None if j==L-1 else j+1

    return fila_arriba, fila_abajo, col_izq, col_der


def indices_primeros_vecinos(i, j, L, is_periodica=False):
    if(j>L-1 or j<0 or i>L-1 or i<0):
        raise ValueError("El valor de los indices debe estar entre 0 y L-1.")

    fila_arriba, fila_abajo, col_izq, col_der = _fila_columna_vecinas(i, j, L, is_periodica)

    return [
        (None, None) if fila_arriba is None else (fila_arriba, j),  # arriba
        (None, None) if col_der is None else (i, col_der),       # derecha
        (None, None) if fila_abajo is None else (fila_abajo, j),    # abajo
        (None, None) if col_izq is None else (i, col_izq),       # izquierda
    ]


def indices_segundos_vecinos(i, j, L, is_periodica=False):
    if(j>L-1 or j<0 or i>L-1 or i<0):
        raise ValueError("El valor de los indices debe estar entre 0 y L-1.")

    fila_arriba, fila_abajo, col_izq, col_der = _fila_columna_vecinas(i, j, L, is_periodica)

    return [
        (None, None) if None in (fila_arriba, col_der) else (fila_arriba, col_der),  # arriba-derecha
        (None, None) if None in (fila_abajo, col_der) else (fila_abajo, col_der),   # abajo-derecha
        (None, None) if None in (fila_abajo, col_izq) else (fila_abajo, col_izq),   # abajo-izquierda
        (None, None) if None in (fila_arriba, col_izq) else (fila_arriba, col_izq),  # arriba-izquierda
    ]
