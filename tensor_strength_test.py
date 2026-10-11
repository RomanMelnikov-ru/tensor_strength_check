import numpy as np
import streamlit as st
import plotly.graph_objs as go
from math import radians

st.set_page_config(
    layout="wide",
    page_title="Анализ НДС",
    initial_sidebar_state="expanded"
)

# Функции для расчетов
def calculate_moduli(E, nu):
    G = E / (2 * (1 + nu))       # Модуль сдвига
    K = E / (3 * (1 - 2 * nu))   # Объемный модуль
    return G, K

def split_stress_tensor(sigma):
    sigma_mean = np.trace(sigma) / 3
    sigma_spherical = sigma_mean * np.eye(3)
    sigma_deviatoric = sigma - sigma_spherical
    return sigma_spherical, sigma_deviatoric

def calculate_strain_tensor(sigma, G, K):
    sigma_spherical, sigma_deviatoric = split_stress_tensor(sigma)
    epsilon_spherical = sigma_spherical / (3 * K)
    epsilon_deviatoric = sigma_deviatoric / (2 * G)
    epsilon = epsilon_spherical + epsilon_deviatoric
    return epsilon_spherical, epsilon_deviatoric, epsilon

def calculate_principal_stresses(sigma):
    eigenvalues, eigenvectors = np.linalg.eig(sigma)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors

# Границы пирамиды Мора-Кулона
def edge1(sigma1, c, phi):
    return sigma1 - (2 * c * np.cos(phi) + 2 * sigma1 * np.sin(phi)) / (1 + np.sin(phi))

def edge2(sigma1, c, phi):
    return sigma1 - (2 * c * np.cos(phi) + 2 * sigma1 * np.sin(phi)) / (1 + np.sin(phi))

def edge3(sigma2, c, phi):
    return sigma2 - (2 * c * np.cos(phi) + 2 * sigma2 * np.sin(phi)) / (1 + np.sin(phi))

def edge4(sigma1, c, phi):
    return sigma1 + (2 * c * np.cos(phi) + 2 * sigma1 * np.sin(phi)) / (1 - np.sin(phi))

def edge5(sigma1, c, phi):
    return sigma1 + (2 * c * np.cos(phi) + 2 * sigma1 * np.sin(phi)) / (1 - np.sin(phi))

def edge6(sigma2, c, phi):
    return sigma2 + (2 * c * np.cos(phi) + 2 * sigma2 * np.sin(phi)) / (1 - np.sin(phi))

def find_intersection(edge_func, sigma_range, plane_constant, vertex, axis, c, phi):
    for sigma in sigma_range:
        if axis == 1:
            sigma3 = edge_func(sigma, c, phi)
            if sigma + sigma + sigma3 >= plane_constant:
                return np.array([sigma, sigma, sigma3])
        elif axis == 2:
            sigma2 = edge_func(sigma, c, phi)
            if sigma + sigma2 + sigma >= plane_constant:
                return np.array([sigma, sigma2, sigma])
        elif axis == 3:
            sigma1 = edge_func(sigma, c, phi)
            if sigma1 + sigma + sigma >= plane_constant:
                return np.array([sigma1, sigma, sigma])
    return None

def update_plot(eigenvalues, c, phi):
    fig = go.Figure()

    # Приводим главные напряжения к обычным float (для JSON-сериализации)
    s1 = float(eigenvalues[0])
    s2 = float(eigenvalues[1])
    s3 = float(eigenvalues[2])

    # Гидростатическая ось
    hydrostatic_axis = np.linspace(-10, 50, 50).tolist()
    fig.add_trace(go.Scatter3d(
        x=hydrostatic_axis, y=hydrostatic_axis, z=hydrostatic_axis,
        mode='lines', line=dict(color='black', dash='dash'),
        name="Гидростатическая ось"
    ))

    # Оси координат
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 40],
                               mode='lines', line=dict(color='red'), name="σ₁"))
    fig.add_trace(go.Scatter3d(x=[0, 40], y=[0, 0], z=[0, 0],
                               mode='lines', line=dict(color='green'), name="σ₂"))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 40], z=[0, 0],
                               mode='lines', line=dict(color='blue'), name="σ₃"))

    # Точка напряжённого состояния
    fig.add_trace(go.Scatter3d(
        x=[s2], y=[s3], z=[s1], mode='markers',
        marker=dict(color='red', size=5), name="Напряженное состояние"
    ))

    # Проекция на гидростатическую ось
    hydro_mean = float(np.mean([s1, s2, s3]))
    fig.add_trace(go.Scatter3d(
        x=[0, hydro_mean], y=[0, hydro_mean], z=[0, hydro_mean],
        mode='lines', line=dict(color='magenta', width=2),
        name="Шаровая часть напряжения"
    ))
    fig.add_trace(go.Scatter3d(
        x=[hydro_mean, s2], y=[hydro_mean, s3], z=[hydro_mean, s1],
        mode='lines', line=dict(color='green', width=2),
        name="Девиаторная часть напряжения"
    ))

    # Пирамида Мора-Кулона
    sigma_vertex = -c / np.tan(phi)
    vertex = np.array([sigma_vertex, sigma_vertex, sigma_vertex])

    intersections = [
        find_intersection(edge1, np.linspace(sigma_vertex, 100, 1000), 150, vertex, 1, c, phi),
        find_intersection(edge2, np.linspace(sigma_vertex, 100, 1000), 150, vertex, 2, c, phi),
        find_intersection(edge3, np.linspace(sigma_vertex, 100, 1000), 150, vertex, 3, c, phi),
        find_intersection(edge4, np.linspace(sigma_vertex, 100, 1000), 150, vertex, 1, c, phi),
        find_intersection(edge5, np.linspace(sigma_vertex, 100, 1000), 150, vertex, 2, c, phi),
        find_intersection(edge6, np.linspace(sigma_vertex, 100, 1000), 150, vertex, 3, c, phi),
    ]

    if all(i is not None for i in intersections):
        all_points = [vertex] + intersections
        x = [float(p[0]) for p in all_points]
        y = [float(p[1]) for p in all_points]
        z = [float(p[2]) for p in all_points]

        i_idx = [0, 0, 0, 0, 0, 0]
        j_idx = [5, 1, 6, 2, 4, 3]
        k_idx = [1, 6, 2, 4, 3, 5]

        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i_idx, j=j_idx, k=k_idx,
            color='orange', opacity=0.2,
            name="Поверхность прочности", showlegend=True
        ))

        edges = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)]
        for idx, e in enumerate(edges):
            fig.add_trace(go.Scatter3d(
                x=[x[e[0]], x[e[1]]],
                y=[y[e[0]], y[e[1]]],
                z=[z[e[0]], z[e[1]]],
                mode='lines',
                line=dict(color='orange', width=2),
                name="Ребра пирамиды" if idx == 0 else None,
                showlegend=(idx == 0),
                legendgroup="edges"
            ))

    fig.update_layout(
        scene=dict(
            xaxis_title='σ₂',
            yaxis_title='σ₃',
            zaxis_title='σ₁',
            xaxis=dict(range=[-50, 150]),
            yaxis=dict(range=[-50, 150]),
            zaxis=dict(range=[-50, 150]),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        title="Анализ напряженно-деформированного состояния",
        height=900,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    st.plotly_chart(fig, use_container_width=True, config={
        'displayModeBar': True,
        'responsive': True
    })

def check_mohr_coulomb_strength(sigma1, sigma2, sigma3, c, phi):
    def strength_condition(si, sj):
        return abs(si - sj) <= 2 * c * np.cos(phi) / (1 - np.sin(phi))
    return (strength_condition(sigma1, sigma2) and
            strength_condition(sigma1, sigma3) and
            strength_condition(sigma2, sigma3))

def calculate():
    try:
        sigma = np.zeros((3, 3))
        sigma[0, 0] = float(st.session_state.sigma11)
        sigma[0, 1] = float(st.session_state.tau12)
        sigma[0, 2] = float(st.session_state.tau13)
        sigma[1, 1] = float(st.session_state.sigma22)
        sigma[1, 2] = float(st.session_state.tau23)
        sigma[2, 2] = float(st.session_state.sigma33)
        sigma[1, 0] = sigma[0, 1]
        sigma[2, 0] = sigma[0, 2]
        sigma[2, 1] = sigma[1, 2]

        E = float(st.session_state.E)
        nu = float(st.session_state.nu)
        c = float(st.session_state.c)
        phi = radians(float(st.session_state.phi))

        G, K = calculate_moduli(E, nu)
        eigenvalues, eigenvectors = calculate_principal_stresses(sigma)
        sigma_spherical, sigma_deviatoric = split_stress_tensor(sigma)
        epsilon_spherical, epsilon_deviatoric, epsilon = calculate_strain_tensor(sigma, G, K)

        strength_check = check_mohr_coulomb_strength(
            float(eigenvalues[0]), float(eigenvalues[1]), float(eigenvalues[2]), c, phi
        )

        def calculate_spherical_and_deviatoric_principal_stresses(ev):
            sigma_mean = np.mean(ev)
            sigma_sph = sigma_mean * np.eye(3)
            sigma_dev = np.diag(ev) - sigma_sph
            return sigma_sph, sigma_dev

        sigma_spherical_principal, sigma_deviatoric_principal = \
            calculate_spherical_and_deviatoric_principal_stresses(eigenvalues)

        with st.expander("Результаты расчета"):
            st.write(f"Модуль сдвига G: {round(G, 1)} кПа")
            st.write(f"Объемный модуль K: {round(K, 1)} кПа\n")

            def print_matrix(matrix, name, precision):
                st.write(f"{name}:\n")
                if precision == 1:
                    fmt = "{:.1f}"
                elif precision == 5:
                    fmt = "{:.5f}"
                else:
                    fmt = "{}"
                st.code(
                    f"{fmt.format(matrix[0,0])}\t{fmt.format(matrix[0,1])}\t{fmt.format(matrix[0,2])}\n"
                    f"{fmt.format(matrix[1,0])}\t{fmt.format(matrix[1,1])}\t{fmt.format(matrix[1,2])}\n"
                    f"{fmt.format(matrix[2,0])}\t{fmt.format(matrix[2,1])}\t{fmt.format(matrix[2,2])}"
                )

            print_matrix(sigma_spherical, "Шаровый тензор напряжений (кПа)", 1)
            print_matrix(sigma_deviatoric, "Тензор-девиатор напряжений (кПа)", 1)

            st.write(f"Главные напряжения (кПа):\n{np.round(eigenvalues, 1)}")

            print_matrix(sigma_spherical_principal, "Шаровая часть главных напряжений (кПа)", 1)
            print_matrix(sigma_deviatoric_principal, "Девиаторная часть главных напряжений (кПа)", 1)

            print_matrix(epsilon_spherical, "Шаровая часть деформаций", 5)
            print_matrix(epsilon_deviatoric, "Девиаторная часть деформаций", 5)
            print_matrix(epsilon, "Полный тензор деформаций", 5)

            if strength_check:
                st.write("\nПрочность обеспечена (точка внутри пирамиды прочности).")
            else:
                st.write("\nПрочность не обеспечена (точка на грани или за пределами пирамиды прочности).")

        update_plot(eigenvalues, c, phi)

    except ValueError:
        st.error("Проверьте введенные данные!")

# Интерфейс
st.sidebar.header("Тензор напряжений")
st.sidebar.number_input("σ₁₁", value=30.0, key="sigma11", on_change=calculate)
st.sidebar.number_input("τ₁₂", value=10.0, key="tau12", on_change=calculate)
st.sidebar.number_input("τ₁₃", value=0.0, key="tau13", on_change=calculate)
st.sidebar.number_input("σ₂₂", value=30.0, key="sigma22", on_change=calculate)
st.sidebar.number_input("τ₂₃", value=0.0, key="tau23", on_change=calculate)
st.sidebar.number_input("σ₃₃", value=30.0, key="sigma33", on_change=calculate)

st.sidebar.header("Механические характеристики")
st.sidebar.number_input("Модуль Юнга E", value=10000.0, key="E", on_change=calculate)
st.sidebar.number_input("Poisson's ratio ν", value=0.3, key="nu", on_change=calculate)
st.sidebar.number_input("Сцепление c", value=10.0, key="c", on_change=calculate)
st.sidebar.number_input("Угол трения φ (градусы)", value=20.0, key="phi", on_change=calculate)

# Первичный расчет
if 'init' not in st.session_state:
    st.session_state.init = True
    calculate()
