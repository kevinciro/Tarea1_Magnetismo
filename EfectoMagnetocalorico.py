"""
Efecto magnetocalorico (EMC), seccion 3.2.2 del enunciado.

Punto 1: calcular Delta_S_M usando integracion numerica de la relacion de
Maxwell (ecuacion 7):

    Delta_S_M(T, h) ~= 1/(2*Delta_T) * [M(T+Delta_T, h) - M(T-Delta_T, h)]

M es la magnetizacion TOTAL del sistema (no Ms), para frontera periodica y
L=24, comparando h=0 (sin campo) y h=0.5 (con campo). Usa directamente los
resultados ya calculados por main_con_h.py (ResultadosConH/
resumen_main_con_h.csv) -- no vuelve a simular nada.
"""

import os
import numpy as np
import pandas as pd


def calcular_delta_S_M(T_lista, M_lista):
    # Diferencias finitas centradas, generalizadas a un grid de T no
    # necesariamente uniforme (a diferencia de la ecuacion (7), que asume
    # un Delta_T constante): Delta_S_M(T_i) = [M(T_{i+1}) - M(T_{i-1})] /
    # (T_{i+1} - T_{i-1}). Esto se reduce exactamente a la ecuacion (7)
    # cuando el espaciado es uniforme (T_{i+1} - T_{i-1} = 2*Delta_T), que
    # es el caso salvo por el ultimo tramo del grid (1.7 -> 2.0).
    #
    # No se puede calcular en los extremos del rango de T (les falta un
    # vecino de un solo lado), asi que esos dos puntos quedan fuera del
    # resultado.
    T = np.asarray(T_lista, dtype=float)
    M = np.asarray(M_lista, dtype=float)

    orden = np.argsort(T)
    T, M = T[orden], M[orden]

    T_interior = T[1:-1]
    delta_S = (M[2:] - M[:-2]) / (T[2:] - T[:-2])

    return T_interior, delta_S


def delta_S_M_desde_resumen(resumen, h):
    # resumen: DataFrame con columnas T, h, M_mean (salida de
    # metricas.resumir_intentos_with_h sobre ResultadosConH). Devuelve
    # (T, Delta_S_M) para el h pedido, ya ordenado por T.
    sub = resumen[resumen["h"] == h].sort_values("T")
    return calcular_delta_S_M(sub["T"].to_numpy(), sub["M_mean"].to_numpy())


def calcular_RCP(T_lista, delta_S_M_lista):
    # RCP (Relative Cooling Power) = |Delta_S_M|_max * FWHM, donde FWHM es
    # el ancho completo a media altura del pico de |Delta_S_M(T)| (punto 4
    # del enunciado). Los cruces de la mitad del maximo se ubican por
    # interpolacion lineal entre los puntos del grid de T (que es discreto
    # y no necesariamente cae justo en el nivel de medio maximo).
    T = np.asarray(T_lista, dtype=float)
    delta_S = np.asarray(delta_S_M_lista, dtype=float)
    orden = np.argsort(T)
    T, delta_S = T[orden], delta_S[orden]
    abs_delta_S = np.abs(delta_S)

    idx_pico = int(np.argmax(abs_delta_S))
    pico = abs_delta_S[idx_pico]
    T_pico = T[idx_pico]
    medio = pico / 2

    def buscar_cruce(paso):
        # recorre el arreglo desde el pico hacia un lado (paso=-1
        # izquierda, paso=+1 derecha) buscando donde abs_delta_S cruza el
        # nivel de medio maximo, interpolando linealmente entre ese punto y
        # el anterior. None si el pico esta en el borde del rango medido
        # (no hay suficientes puntos para encontrar el cruce).
        i = idx_pico
        while 0 <= i + paso < len(T):
            j = i + paso
            if (abs_delta_S[i] - medio) * (abs_delta_S[j] - medio) <= 0 and abs_delta_S[i] != abs_delta_S[j]:
                frac = (medio - abs_delta_S[i]) / (abs_delta_S[j] - abs_delta_S[i])
                return T[i] + frac * (T[j] - T[i])
            i = j
        return None

    T_izq = buscar_cruce(-1)
    T_der = buscar_cruce(+1)

    fwhm = (T_der - T_izq) if (T_izq is not None and T_der is not None) else None
    rcp = (pico * fwhm) if fwhm is not None else None

    return {
        "T_pico": T_pico, "pico_abs_delta_S_M": pico,
        "T_izq": T_izq, "T_der": T_der, "FWHM": fwhm, "RCP": rcp,
    }


def calcular_tabla_delta_S_M(resumen):
    # Aplica delta_S_M_desde_resumen a cada h presente en resumen y arma un
    # unico DataFrame (columnas T, h, delta_S_M) con todos los resultados,
    # listo para guardar en CSV y graficar despues.
    filas = []
    for h in sorted(resumen["h"].unique()):
        T, delta_S = delta_S_M_desde_resumen(resumen, h)
        for t, ds in zip(T, delta_S):
            filas.append({"T": t, "h": h, "delta_S_M": ds})
    return pd.DataFrame(filas)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from graficas import graficar_delta_S_M

    CARPETA_BASE = "ResultadosConH"
    CARPETA_GRAFICAS = os.path.join(CARPETA_BASE, "graficas")
    ARCHIVO_RESUMEN = os.path.join(CARPETA_BASE, "resumen_main_con_h.csv")
    ARCHIVO_SALIDA = os.path.join(CARPETA_BASE, "delta_S_M.csv")

    os.makedirs(CARPETA_GRAFICAS, exist_ok=True)

    resumen = pd.read_csv(ARCHIVO_RESUMEN)
    tabla = calcular_tabla_delta_S_M(resumen)
    tabla.to_csv(ARCHIVO_SALIDA, index=False)
    print(tabla.to_string(index=False))
    print(f"\nGuardado en {ARCHIVO_SALIDA}")

    h_lista = sorted(resumen["h"].unique())
    graficar_delta_S_M(tabla, h_lista, titulo=r"$\Delta S_M(T)$")
    archivo_grafica = os.path.join(CARPETA_GRAFICAS, "delta_S_M_T.png")
    plt.savefig(archivo_grafica, dpi=110)
    plt.close()
    print(f"Grafica guardada en {archivo_grafica}")

    # RCP (punto 4) para h=0.5, via FWHM del pico de |Delta_S_M|
    H_RCP = 0.5
    sub_h = tabla[tabla["h"] == H_RCP].sort_values("T")
    rcp_info = calcular_RCP(sub_h["T"].to_numpy(), sub_h["delta_S_M"].to_numpy())
    print(f"\n--- RCP para h={H_RCP:g} ---")
    for k, v in rcp_info.items():
        print(f"  {k}: {v}")

    pd.DataFrame([{"h": H_RCP, **rcp_info}]).to_csv(
        os.path.join(CARPETA_BASE, "RCP.csv"), index=False)

    # Grafica con el pico, la mitad del maximo y el ancho FWHM marcados
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sub_h["T"], sub_h["delta_S_M"].abs(), "o--", color="tab:red",
            linewidth=1, markersize=5, label=f"h={H_RCP:g}")
    ax.axhline(rcp_info["pico_abs_delta_S_M"], color="0.5", linestyle=":", linewidth=1)
    ax.axhline(rcp_info["pico_abs_delta_S_M"] / 2, color="0.5", linestyle=":", linewidth=1,
               label="medio maximo")
    if rcp_info["T_izq"] is not None and rcp_info["T_der"] is not None:
        ax.axvspan(rcp_info["T_izq"], rcp_info["T_der"], color="tab:red", alpha=0.15,
                   label=f"FWHM = {rcp_info['FWHM']:.3f}")
    ax.set_xlabel("T")
    ax.set_ylabel(r"$|\Delta S_M|$")
    ax.set_title(f"RCP (h={H_RCP:g}) = {rcp_info['RCP']:.2f}")
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.7, color="0.6", alpha=0.8)
    plt.tight_layout()
    archivo_rcp = os.path.join(CARPETA_GRAFICAS, "RCP_h05.png")
    plt.savefig(archivo_rcp, dpi=110)
    plt.close()
    print(f"Grafica de RCP guardada en {archivo_rcp}")
