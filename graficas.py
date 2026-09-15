# Graficas realizadas por la IA al 100%

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


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

def graficar_historial_with_h(mcs_steps, E_prom, E2_prom, M_prom, M2_prom):

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 14))

    axs[0].plot(mcs_steps, E_prom)
    axs[0].set_ylabel(r"$\langle E \rangle$")

    axs[1].plot(mcs_steps, E2_prom)
    axs[1].set_ylabel(r"$\langle E^2 \rangle$")

    axs[2].plot(mcs_steps, M_prom)
    axs[2].set_ylabel(r"$\langle M \rangle$")

    axs[3].plot(mcs_steps, M2_prom)
    axs[3].set_ylabel(r"$\langle M^2 \rangle$")
    axs[3].set_xlabel("MCS")

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

def graficar_historial_normalizado_with_h(mcs_steps, E_prom, E2_prom, M_prom, M2_prom):
    def normalizar(x):
        x = np.array(x, dtype=float)
        return (x - x.min()) / (x.max() - x.min())

    plt.plot(mcs_steps, normalizar(E_prom), label=r"$\langle E \rangle$")
    plt.plot(mcs_steps, normalizar(E2_prom), label=r"$\langle E^2 \rangle$")
    plt.plot(mcs_steps, normalizar(M_prom), label=r"$\langle M \rangle$")
    plt.plot(mcs_steps, normalizar(M2_prom), label=r"$\langle M^2 \rangle$")
    plt.xlabel("MCS")
    plt.ylabel("Valor normalizado")
    plt.legend()
    plt.show()

def graficar_observable_paneles(resumen, columna, ylabel, titulo=None, T_min=None, T_max=None):
    # Un panel por condicion de frontera (libre, periodica), cada uno con
    # los puntos (scatter) de L=16 (azul) y L=24 (roja) con su barra de
    # error (SEM sobre los intentos), unidos por una linea punteada como
    # guia visual. T_min/T_max permiten hacer zoom a una ventana de
    # temperatura, igual que el zoom de U(T) cerca de la transicion.
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
            ax.errorbar(sub["T"], sub[f"{columna}_mean"], yerr=sub[f"{columna}_sem"],
                        fmt="o", color=colores_L.get(L), linestyle="--", linewidth=1,
                        markersize=5, capsize=3, label=f"L={L}")

        ax.set_xlabel("T")
        ax.set_title(f"Frontera {nombre}")
        ax.legend()
        ax.grid(True, linestyle="--", linewidth=0.7, color="0.6", alpha=0.8)

    axs[0].set_ylabel(ylabel)
    if titulo:
        fig.suptitle(titulo)
    plt.tight_layout()
    plt.show()

def graficar_observable_h(resumen, columna, ylabel, h_lista, titulo=None, T_min=None, T_max=None):
    # Un solo panel para barridos con campo externo (columna "h" en
    # resumen): una linea por cada h en h_lista (con su barra de error), asi
    # se comparan directamente en el mismo eje T. Si resumen tiene mas de un
    # L, se grafica una linea por cada combinacion (L, h).
    sub = resumen
    if T_min is not None:
        sub = sub[sub["T"] >= T_min]
    if T_max is not None:
        sub = sub[sub["T"] <= T_max]

    L_unicos = sorted(sub["L"].unique())
    # Paleta fija de colores oscuros/opacos (en vez de un colormap como
    # viridis, cuyo extremo amarillo queda muy chillon), suficiente para
    # unos pocos valores de h; si hay mas de 6 se repiten.
    paleta = ["tab:blue", "tab:red", "tab:green", "tab:purple", "tab:brown", "tab:gray"]
    colores_h = {h: paleta[i % len(paleta)] for i, h in enumerate(sorted(h_lista))}

    fig, ax = plt.subplots(figsize=(8, 5))
    for h in h_lista:
        sub_h = sub[sub["h"] == h]
        for L in L_unicos:
            sub_hl = sub_h[sub_h["L"] == L].sort_values("T")
            etiqueta = f"h={h:g}" if len(L_unicos) == 1 else f"L={L}, h={h:g}"
            ax.errorbar(sub_hl["T"], sub_hl[f"{columna}_mean"], yerr=sub_hl[f"{columna}_sem"],
                        fmt="o", color=colores_h[h], linestyle="--", linewidth=1,
                        markersize=5, capsize=3, label=etiqueta)

    ax.set_xlabel("T")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.7, color="0.6", alpha=0.8)
    if titulo:
        ax.set_title(titulo)
    plt.tight_layout()
    plt.show()

def generar_video(cuadros, archivo_salida, fps=1):
    # cuadros: lista de (T, S, Ms) de mayor a menor T, una instantanea de la
    # red por cada temperatura visitada (ver evolucion_gif.py / MonteCarlo._correr_una_cadena).
    # Video con dos paneles: izquierda, la curva completa (fija) de Ms vs T
    # con un punto/flecha que señala el cuadro actual; derecha, la
    # configuracion de la red como flechas (+1 roja arriba, -1 azul abajo),
    # sobre una cuadricula sin relleno de color.
    cmap = ListedColormap(["tab:blue", "tab:red"])  # -1 -> azul, +1 -> rojo
    norm = BoundaryNorm([-1.5, 0, 1.5], cmap.N)  # separa limpio -1 de +1, sin interpolar colores

    _, S0, Ms0 = cuadros[0]
    L_local = S0.shape[0]

    # La curva completa de Ms vs T se conoce de antemano (los cuadros ya
    # vienen completos), asi que se dibuja fija una sola vez; solo el
    # punto/flecha que señala "aqui vamos" se mueve cuadro a cuadro.
    T_todos = [c[0] for c in cuadros]
    Ms_todos = [c[2] for c in cuadros]

    # Posiciones (X, Y) del centro de cada celda. Y se calcula "invertido"
    # (L_local-1-fila) en vez de usar invert_yaxis(), para que la fila 0
    # quede arriba en la figura SIN que eso invierta tambien el sentido de
    # las flechas (con invert_yaxis, una flecha "hacia +Y" terminaria
    # apuntando hacia abajo en pantalla, que es justo lo contrario de lo
    # que se quiere para spin=+1).
    columnas, filas = np.meshgrid(np.arange(L_local), np.arange(L_local))
    X = columnas.astype(float)
    Y = (L_local - 1 - filas).astype(float)
    U = np.zeros_like(S0, dtype=float)  # las flechas son verticales, sin componente horizontal

    fig, (ax_izq, ax_der) = plt.subplots(1, 2, figsize=(12, 6), dpi=100)

    # --- Panel izquierdo: Ms vs T, curva completa y fija ---
    ax_izq.plot(T_todos, Ms_todos, "o-", color="0.3", markersize=4, linewidth=1.5)
    ax_izq.set_xlabel("T")
    ax_izq.set_ylabel(r"$M_s$")
    ax_izq.set_ylim(-0.03, 0.55)
    # Eje invertido: T alta a la izquierda, T baja a la derecha, para que el
    # punto avance de izquierda a derecha a medida que el video enfria el
    # sistema (mismo orden cronologico que T_todos, que viene de mayor a menor).
    ax_izq.set_xlim(T_todos[0] * 1.05, T_todos[-1] - 0.05 * T_todos[0])
    ax_izq.invert_xaxis()
    ax_izq.set_xticks(np.arange(0.0, T_todos[0] + 0.01, 0.2))

    punto_actual, = ax_izq.plot([T_todos[0]], [Ms0], "o", color="tab:red", markersize=10, zorder=5)

    # --- Panel derecho: configuracion de la red (flechas por spin) ---
    ax_der.set_aspect("equal")
    ax_der.set_xlim(-0.5, L_local - 0.5)
    ax_der.set_ylim(-0.5, L_local - 0.5)

    # Solo el perimetro de cada celda (cuadricula), sin relleno de color
    ax_der.set_xticks(np.arange(-0.5, L_local, 1))
    ax_der.set_yticks(np.arange(-0.5, L_local, 1))
    ax_der.set_xticklabels([])
    ax_der.set_yticklabels([])
    ax_der.tick_params(length=0)
    ax_der.grid(True, color="black", linewidth=0.5)

    V0 = 0.8 * S0  # largo de la flecha (menor a 1 celda para que no se toquen entre si)
    # width mas chico (linea/asta mas delgada) y headwidth/headlength/
    # headaxislength mas chicos (punta mas pequeña) que el default de
    # matplotlib (width=0.005 del ancho del eje, head 3/5/4.5).
    q = ax_der.quiver(X, Y, U, V0, S0, cmap=cmap, norm=norm,
                       pivot="middle", scale=1, scale_units="xy", width=0.015,
                       headwidth=2.5, headlength=2.5, headaxislength=1.8)

    titulo = ax_der.set_title(f"T = {cuadros[0][0]:.2f}   Ms = {Ms0:.3f}")

    fig.tight_layout()

    # Se renderiza cada cuadro con matplotlib y se escribe directo a un
    # video con OpenCV (ffmpeg no esta instalado, que es lo que
    # matplotlib.animation necesitaria para exportar video).
    fig.canvas.draw()
    ancho, alto = fig.canvas.get_width_height()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    escritor = cv2.VideoWriter(archivo_salida, fourcc, fps, (ancho, alto))

    def escribir_cuadro():
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frame_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        escritor.write(frame_bgr)

    escribir_cuadro()  # primer cuadro (ya dibujado arriba)
    for i in range(1, len(cuadros)):
        t, S_i, Ms_i = cuadros[i]

        q.set_UVC(U, 0.8 * S_i, S_i)
        titulo.set_text(f"T = {t:.2f}   Ms = {Ms_i:.3f}")

        punto_actual.set_data([t], [Ms_i])

        escribir_cuadro()

    escritor.release()
    plt.close(fig)
