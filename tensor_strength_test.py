import numpy as np
import streamlit as st
import plotly.graph_objs as go
from math import radians

st.set_page_config(
    layout="wide",
    page_title="Анализ НДС",
    initial_sidebar_state="expanded"
)

# ---------- Функции расчёта ----------
def calculate_moduli(E, nu):
    G = E / (2 * (1 + nu))
    K = E / (3 * (1 - 2 * nu))
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
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors

def calculate_spherical_and_deviatoric_principal_stresses(eigenvalues):
    eigenvalues = np.real(eigenvalues).astype(float)
    sigma_mean = float(np.mean(eigenvalues))
    sigma_spherical = sigma_mean * np.eye(3)
    sigma_deviatoric = np.diag(eigenvalues) - sigma_spherical
    return sigma_spherical, sigma_deviatoric

def check_mohr_coulomb_strength(sigma1, sigma2, sigma3, c, phi):
    def strength_condition(si, sj):
        return abs(si - sj) <= 2 * c * np.cos(phi) / (1 - np.sin(phi))
    return (strength_condition(sigma1, sigma2) and
            strength_condition(sigma1, sigma3) and
            strength_condition(sigma2, sigma3))

# ---------- Границы пирамиды Мора-Кулона ----------
def edge1(s1, c, phi):
    return s1 - (2*c*np.cos(phi) + 2*s1*np.sin(phi)) / (1 + np.sin(phi))

def edge2(s1, c, phi):
    return s1 - (2*c*np.cos(phi) + 2*s1*np.sin(phi)) / (1 + np.sin(phi))

def edge3(s2, c, phi):
    return s2 - (2*c*np.cos(phi) + 2*s2*np.sin(phi)) / (1 + np.sin(phi))

def edge4(s1, c, phi):
    return s1 + (2*c*np.cos(phi) + 2*s1*np.sin(phi)) / (1 - np.sin(phi))

def edge5(s1, c, phi):
    return s1 + (2*c*np.cos(phi) + 2*s1*np.sin(phi)) / (1 - np.sin(phi))

def edge6(s2, c, phi):
    return s2 + (2*c*np.cos(phi) + 2*s2*np.sin(phi)) / (1 - np.sin(phi))

def find_intersection(edge_func, sigma_range, plane_constant, vertex, axis, c, phi):
    for sigma in sigma_range:
        if axis == 1:
            s3 = edge_func(sigma, c, phi)
            if sigma + sigma + s3 >= plane_constant:
                return np.array([sigma, sigma, s3])
        elif axis == 2:
            s2 = edge_func(sigma, c, phi)
            if sigma + s2 + sigma >= plane_constant:
                return np.array([sigma, s2, sigma])
        elif axis == 3:
            s1 = edge_func(sigma, c, phi)
            if s1 + sigma + sigma >= plane_constant:
                return np.array([s1, sigma, sigma])
    return None

# ---------- График ----------
def update_plot(eigenvalues, c, phi, show_pyramid):
    fig = go.Figure()

    s1, s2, s3 = float(eigenvalues[0]), float(eigenvalues[1]), float(eigenvalues[2])

    hydrostatic_axis = np.linspace(-10, 50, 50).tolist()
    fig.add_trace(go.Scatter3d(
        x=hydrostatic_axis, y=hydrostatic_axis, z=hydrostatic_axis,
        mode='lines', line=dict(color='black', dash='dash'),
        name="Гидростатическая ось"
    ))

    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 40],
                               mode='lines', line=dict(color='red'), name="σ₁"))
    fig.add_trace(go.Scatter3d(x=[0, 40], y=[0, 0], z=[0, 0],
                               mode='lines', line=dict(color='green'), name="σ₂"))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 40], z=[0, 0],
                               mode='lines', line=dict(color='blue'), name="σ₃"))

    fig.add_trace(go.Scatter3d(
        x=[s2], y=[s3], z=[s1], mode='markers',
        marker=dict(color='red', size=5), name="Напряженное состояние"
    ))

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

    if show_pyramid:
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
                name="Поверхность прочности", showlegend=True,
                legendgroup="pyramid"
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
                    legendgroup="pyramid"
                ))

    fig.update_layout(
        uirevision="constant",
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

# ---------- Основной расчёт ----------
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
        eigenvalues, _ = calculate_principal_stresses(sigma)
        eigenvalues = np.real(eigenvalues).astype(float)

        sigma_spherical, sigma_deviatoric = split_stress_tensor(sigma)
        epsilon_spherical, epsilon_deviatoric, epsilon = calculate_strain_tensor(sigma, G, K)

        strength_check = check_mohr_coulomb_strength(
            float(eigenvalues[0]), float(eigenvalues[1]), float(eigenvalues[2]), c, phi
        )

        sigma_spherical_principal, sigma_deviatoric_principal = \
            calculate_spherical_and_deviatoric_principal_stresses(eigenvalues)

        # ---------- Гармошка «Результаты расчета» ----------
        with st.expander("Результаты расчета", expanded=True):
            st.write(f"Модуль сдвига G: {round(G, 1)} кПа")
            st.write(f"Объемный модуль K: {round(K, 1)} кПа\n")

            def print_matrix(matrix, name, precision):
                st.write(f"{name}:\n")
                matrix = np.real(matrix)
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

            st.write(f"Главные напряжения (кПа): {np.round(eigenvalues, 1).tolist()}")

            print_matrix(sigma_spherical_principal, "Шаровая часть главных напряжений (кПа)", 1)
            print_matrix(sigma_deviatoric_principal, "Девиаторная часть главных напряжений (кПа)", 1)

            print_matrix(epsilon_spherical, "Шаровая часть деформаций", 5)
            print_matrix(epsilon_deviatoric, "Девиаторная часть деформаций", 5)
            print_matrix(epsilon, "Полный тензор деформаций", 5)

            if strength_check:
                st.success("Прочность обеспечена (точка внутри пирамиды прочности).")
            else:
                st.error("Прочность не обеспечена (точка на грани или за пределами пирамиды).")

        # ---------- Гармошка «Формулы для расчета» ----------
        with st.expander("Формулы для расчета", expanded=False):
            st.markdown(r"""
### 1. Упругие константы

$$
G = \dfrac{E}{2(1+\nu)}, \qquad
K = \dfrac{E}{3(1-2\nu)}
$$

### 2. Разложение тензора напряжений

Среднее (гидростатическое) напряжение:

$$
\sigma_m = \dfrac{1}{3}\,\text{tr}\,\boldsymbol{\sigma} = \dfrac{\sigma_{11} + \sigma_{22} + \sigma_{33}}{3}
$$

Шаровая часть:

$$
\boldsymbol{\sigma}^{sph} = \sigma_m \,\mathbf{I}
$$

Девиаторная часть:

$$
\boldsymbol{\sigma}^{dev} = \boldsymbol{\sigma} - \boldsymbol{\sigma}^{sph}
$$

### 3. Главные напряжения

Решается характеристическое уравнение:

$$
\det(\boldsymbol{\sigma} - \lambda \mathbf{I}) = 0
\quad\Longrightarrow\quad
\lambda_1 \ge \lambda_2 \ge \lambda_3
$$

Главные напряжения упорядочиваются по убыванию:
$\sigma_1 = \lambda_1,\ \sigma_2 = \lambda_2,\ \sigma_3 = \lambda_3$.

### 4. Тензор деформаций (закон Гука)

Шаровая часть:

$$
\boldsymbol{\varepsilon}^{sph} = \dfrac{\boldsymbol{\sigma}^{sph}}{3K}
$$

Девиаторная часть:

$$
\boldsymbol{\varepsilon}^{dev} = \dfrac{\boldsymbol{\sigma}^{dev}}{2G}
$$

Полный тензор деформаций:

$$
\boldsymbol{\varepsilon} = \boldsymbol{\varepsilon}^{sph} + \boldsymbol{\varepsilon}^{dev}
= \dfrac{1}{3K}\boldsymbol{\sigma}^{sph} + \dfrac{1}{2G}\boldsymbol{\sigma}^{dev}
$$

### 5. Критерий прочности Мора – Кулона

Условие прочности для каждой пары главных напряжений:

$$
|\sigma_i - \sigma_j| \le \dfrac{2c\,\cos\varphi}{1 - \sin\varphi}
\qquad (i \ne j)
$$

где:
- $c$ — сцепление,
- $\varphi$ — угол внутреннего трения.

Прочность обеспечена, если неравенство выполняется для всех трёх пар:
$(\sigma_1,\sigma_2)$, $(\sigma_1,\sigma_3)$, $(\sigma_2,\sigma_3)$.

### 6. Пирамида Мора – Кулона

Вершина пирамиды на гидростатической оси:

$$
\sigma_{vertex} = -\dfrac{c}{\tan\varphi}
$$

Ребра пирамиды описываются уравнениями вида:

$$
\sigma_1 = \dfrac{2c\,\cos\varphi + 2\sigma_3\,\sin\varphi}{1 - \sin\varphi}
\quad\text{(для грани } \sigma_1 \ge \sigma_2 = \sigma_3\text{)}
$$

и аналогичными соотношениями для остальных пяти рёбер, полученных циклической перестановкой индексов.
""")

        # ---------- График ----------
        show_pyramid = st.session_state.get("show_pyramid", False)
        update_plot(eigenvalues, c, phi, show_pyramid)

    except (ValueError, ZeroDivisionError) as e:
        st.error(f"Проверьте введенные данные! ({e})")

# ---------- Синхронизация симметричных компонент ----------
def sync_symmetric():
    st.session_state.tau21 = st.session_state.tau12
    st.session_state.tau31 = st.session_state.tau13
    st.session_state.tau32 = st.session_state.tau23
    calculate()

# ---------- Инициализация зеркальных компонент ----------
if 'tau21' not in st.session_state:
    st.session_state.tau21 = st.session_state.get('tau12', 0.0)
if 'tau31' not in st.session_state:
    st.session_state.tau31 = st.session_state.get('tau13', 0.0)
if 'tau32' not in st.session_state:
    st.session_state.tau32 = st.session_state.get('tau23', 0.0)

# ---------- Интерфейс ----------
st.sidebar.header("Тензор напряжений")

c1, c2, c3 = st.sidebar.columns(3)
c1.number_input("σ₁₁", value=30.0, key="sigma11", on_change=calculate)
c2.number_input("τ₁₂", value=0.0, key="tau12", on_change=sync_symmetric)
c3.number_input("τ₁₃", value=0.0, key="tau13", on_change=sync_symmetric)

c1, c2, c3 = st.sidebar.columns(3)
c1.number_input("τ₂₁", key="tau21", disabled=True)
c2.number_input("σ₂₂", value=30.0, key="sigma22", on_change=calculate)
c3.number_input("τ₂₃", value=0.0, key="tau23", on_change=sync_symmetric)

c1, c2, c3 = st.sidebar.columns(3)
c1.number_input("τ₃₁", key="tau31", disabled=True)
c2.number_input("τ₃₂", key="tau32", disabled=True)
c3.number_input("σ₃₃", value=30.0, key="sigma33", on_change=calculate)

st.sidebar.header("Механические характеристики")
st.sidebar.number_input("Модуль Юнга E (кПа)", value=10000.0, key="E", on_change=calculate)
st.sidebar.number_input("Коэффициент Пуассона ν", value=0.3, key="nu", on_change=calculate)
st.sidebar.number_input("Сцепление c (кПа)", value=10.0, key="c", on_change=calculate)
st.sidebar.number_input("Угол трения φ (°)", value=20.0, key="phi", on_change=calculate)

st.sidebar.header("Отображение")
st.sidebar.checkbox(
    "Показать поверхность прочности",
    value=False,
    key="show_pyramid",
    on_change=calculate
)

# ---------- Первичный расчёт ----------
if 'init' not in st.session_state:
    st.session_state.init = True
    calculate()
