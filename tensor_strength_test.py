import numpy as np
import streamlit as st
import plotly.graph_objs as go
from math import radians

# Глобальная переменная для хранения состояния камеры
camera_state = None

# Функции для расчетов
def calculate_moduli(E, nu):
    G = E / (2 * (1 + nu))  # Модуль сдвига
    K = E / (3 * (1 - 2 * nu))  # Объемный модуль
    return G, K

def split_stress_tensor(sigma):
    sigma_mean = np.trace(sigma) / 3  # Среднее напряжение
    sigma_spherical = sigma_mean * np.eye(3)  # Шаровая часть
    sigma_deviatoric = sigma - sigma_spherical  # Девиаторная часть
    return sigma_spherical, sigma_deviatoric

def calculate_strain_tensor(sigma, G, K):
    sigma_spherical, sigma_deviatoric = split_stress_tensor(sigma)
    epsilon_spherical = sigma_spherical / (3 * K)  # Шаровая часть деформаций
    epsilon_deviatoric = sigma_deviatoric / (2 * G)  # Девиаторная часть деформаций
    epsilon = epsilon_spherical + epsilon_deviatoric  # Полный тензор деформаций
    return epsilon_spherical, epsilon_deviatoric, epsilon

def calculate_principal_stresses(sigma):
    eigenvalues, eigenvectors = np.linalg.eig(sigma)  # Главные напряжения и направления
    idx = eigenvalues.argsort()[::-1]  # Сортировка по убыванию
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors

# Границы пирамиды Мора-Кулона
def edge1(sigma1, c, phi):  # sigma1 = sigma2 > sigma3
    return sigma1 - (2 * c * np.cos(phi) + 2 * sigma1 * np.sin(phi)) / (1 + np.sin(phi))

def edge2(sigma1, c, phi):  # sigma1 = sigma3 > sigma2
    return sigma1 - (2 * c * np.cos(phi) + 2 * sigma1 * np.sin(phi)) / (1 + np.sin(phi))

def edge3(sigma2, c, phi):  # sigma2 = sigma3 > sigma1
    return sigma2 - (2 * c * np.cos(phi) + 2 * sigma2 * np.sin(phi)) / (1 + np.sin(phi))

def edge4(sigma1, c, phi):  # sigma1 = sigma2 < sigma3
    return sigma1 + (2 * c * np.cos(phi) + 2 * sigma1 * np.sin(phi)) / (1 - np.sin(phi))

def edge5(sigma1, c, phi):  # sigma1 = sigma3 < sigma2
    return sigma1 + (2 * c * np.cos(phi) + 2 * sigma1 * np.sin(phi)) / (1 - np.sin(phi))

def edge6(sigma2, c, phi):  # sigma2 = sigma3 < sigma1
    return sigma2 + (2 * c * np.cos(phi) + 2 * sigma2 * np.sin(phi)) / (1 - np.sin(phi))

# Поиск пересечения ребра с плоскостью
def find_intersection(edge_func, sigma_range, plane_constant, vertex, axis, c, phi):
    for sigma in sigma_range:
        if axis == 1:  # sigma1 = sigma2 > sigma3 или sigma1 = sigma2 < sigma3
            sigma3 = edge_func(sigma, c, phi)
            if sigma + sigma + sigma3 >= plane_constant:
                return np.array([sigma, sigma, sigma3])
        elif axis == 2:  # sigma1 = sigma3 > sigma2 или sigma1 = sigma3 < sigma2
            sigma2 = edge_func(sigma, c, phi)
            if sigma + sigma2 + sigma >= plane_constant:
                return np.array([sigma, sigma2, sigma])
        elif axis == 3:  # sigma2 = sigma3 > sigma1 или sigma2 = sigma3 < sigma1
            sigma1 = edge_func(sigma, c, phi)
            if sigma1 + sigma + sigma >= plane_constant:
                return np.array([sigma1, sigma, sigma])
    return None

# Обновление графика
def update_plot(eigenvalues, c, phi):
    global camera_state  # Используем глобальную переменную для камеры

    fig = go.Figure()

    # Гидростатическая ось
    hydrostatic_axis = np.linspace(-10, 50, 50)
    fig.add_trace(go.Scatter3d(
        x=hydrostatic_axis, y=hydrostatic_axis, z=hydrostatic_axis,
        mode='lines', line=dict(color='black', dash='dash'), name="Гидростатическая ось"
    ))

    # Оси координат (σ₁ вертикальная)
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 40], mode='lines', line=dict(color='red'), name="σ₁"))
    fig.add_trace(go.Scatter3d(x=[0, 40], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='green'), name="σ₂"))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 40], z=[0, 0], mode='lines', line=dict(color='blue'), name="σ₃"))

    # Точка напряженного состояния
    fig.add_trace(go.Scatter3d(
        x=[eigenvalues[1]], y=[eigenvalues[2]], z=[eigenvalues[0]], mode='markers',
        marker=dict(color='red', size=5), name="Напряженное состояние"
    ))

    # Проекция точки на гидростатическую ось
    hydrostatic_projection = np.mean(eigenvalues) * np.ones(3)
    fig.add_trace(go.Scatter3d(
        x=[0, hydrostatic_projection[1]], y=[0, hydrostatic_projection[2]],
        z=[0, hydrostatic_projection[0]], mode='lines', line=dict(color='magenta', width=2),
        name="Шаровая часть напряжения"
    ))

    fig.add_trace(go.Scatter3d(
        x=[hydrostatic_projection[1], eigenvalues[1]],
        y=[hydrostatic_projection[2], eigenvalues[2]],
        z=[hydrostatic_projection[0], eigenvalues[0]], mode='lines',
        line=dict(color='green', width=2), name="Девиаторная часть напряжения"
    ))

    # Построение пирамиды Мора-Кулона
    sigma_vertex = -c / np.tan(phi)
    vertex = np.array([sigma_vertex, sigma_vertex, sigma_vertex])

    intersections = [
        find_intersection(edge1, np.linspace(sigma_vertex, 100, 1000), 150, vertex, axis=1, c=c, phi=phi),
        find_intersection(edge2, np.linspace(sigma_vertex, 100, 1000), 150, vertex, axis=2, c=c, phi=phi),
        find_intersection(edge3, np.linspace(sigma_vertex, 100, 1000), 150, vertex, axis=3, c=c, phi=phi),
        find_intersection(edge4, np.linspace(sigma_vertex, 100, 1000), 150, vertex, axis=1, c=c, phi=phi),
        find_intersection(edge5, np.linspace(sigma_vertex, 100, 1000), 150, vertex, axis=2, c=c, phi=phi),
        find_intersection(edge6, np.linspace(sigma_vertex, 100, 1000), 150, vertex, axis=3, c=c, phi=phi)
    ]

    if all(intersection is not None for intersection in intersections):
        all_points = [vertex] + intersections
        x = [point[0] for point in all_points]
        y = [point[1] for point in all_points]
        z = [point[2] for point in all_points]

        # Индексы для граней
        i = [0, 0, 0, 0, 0, 0]  # Вершина пирамиды
        j = [5, 1, 6, 2, 4, 3]  # Первая точка грани
        k = [1, 6, 2, 4, 3, 5]  # Вторая точка грани

        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color='orange',
            opacity=0.2,
            reversescale=True,
            name="Поверхность прочности",
            showlegend=True
        ))

        # Ребра пирамиды
        edges = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)]
        for idx, edge in enumerate(edges):
            fig.add_trace(go.Scatter3d(
                x=[x[edge[0]], x[edge[1]]],
                y=[y[edge[0]], y[edge[1]]],
                z=[z[edge[0]], z[edge[1]]],
                mode='lines',
                line=dict(color='orange', width=2),
                name="Ребра пирамиды" if idx == 0 else None,  # Подпись только для первого ребра
                showlegend=True if idx == 0 else False,  # Отображение в легенде только для первого ребра
                legendgroup="edges"  # Группировка рёбер
            ))

    # Настройки графика
    fig.update_layout(scene=dict(
        xaxis_title='σ₂',
        yaxis_title='σ₃',
        zaxis_title='σ₁',
        xaxis=dict(range=[-50, 150]),
        yaxis=dict(range=[-50, 150]),
        zaxis=dict(range=[-50, 150])
    ), title="Анализ напряженно-деформированного состояния")

    # Сохраняем текущее состояние камеры, если оно еще не сохранено
    if camera_state is not None:
        fig.update_layout(scene_camera=camera_state)

    # Сохраняем текущее состояние камеры после отображения графика
    camera_state = fig.layout.scene.camera

    st.plotly_chart(fig)

# Расчет и вывод результатов
def calculate():
    try:
        # Ввод данных
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

        # Расчет модулей
        G, K = calculate_moduli(E, nu)

        # Расчет главных напряжений
        eigenvalues, eigenvectors = calculate_principal_stresses(sigma)

        # Шаровый и девиаторный тензоры
        sigma_spherical, sigma_deviatoric = split_stress_tensor(sigma)

        # Деформации
        epsilon_spherical, epsilon_deviatoric, epsilon = calculate_strain_tensor(sigma, G, K)

        # Проверка прочности по Мора-Кулону
        strength_check = check_mohr_coulomb_strength(eigenvalues[0], eigenvalues[1], eigenvalues[2], c, phi)

        # Расчет шаровой части главных напряжений
        sigma_mean = np.mean(eigenvalues)  # Среднее значение главных напряжений

        # Расчет шаровой и девиаторной части главных напряжений
        def calculate_spherical_and_deviatoric_principal_stresses(eigenvalues):
            sigma_mean = np.mean(eigenvalues)  # Среднее значение главных напряжений
            sigma_spherical = sigma_mean * np.eye(3)  # Шаровая часть
            sigma_deviatoric = np.diag(eigenvalues) - sigma_spherical  # Девиаторная часть
            return sigma_spherical, sigma_deviatoric

        # После расчета главных напряжений
        eigenvalues, eigenvectors = calculate_principal_stresses(sigma)
        sigma_spherical_principal, sigma_deviatoric_principal = calculate_spherical_and_deviatoric_principal_stresses(
            eigenvalues)

        # Вывод результатов
        with st.expander("Результаты расчета"):
            st.write(f"Модуль сдвига G: {round(G, 1)} кПа")
            st.write(f"Объемный модуль K: {round(K, 1)} кПа\n")

            # Вывод матриц как 3×3
            def print_matrix(matrix, name, precision):
                st.write(f"{name}:\n")
                if precision == 1:
                    fmt = "{:.1f}"
                elif precision == 5:
                    fmt = "{:.5f}"
                else:
                    fmt = "{}"
                st.code(f"{fmt.format(matrix[0, 0])}\t{fmt.format(matrix[0, 1])}\t{fmt.format(matrix[0, 2])}\n"
                        f"{fmt.format(matrix[1, 0])}\t{fmt.format(matrix[1, 1])}\t{fmt.format(matrix[1, 2])}\n"
                        f"{fmt.format(matrix[2, 0])}\t{fmt.format(matrix[2, 1])}\t{fmt.format(matrix[2, 2])}")

            # Вывод матриц напряжений с точностью до 1 знака после запятой
            print_matrix(sigma_spherical, "Шаровый тензор напряжений (кПа)", 1)
            print_matrix(sigma_deviatoric, "Тензор-девиатор напряжений (кПа)", 1)

            st.write(f"Главные напряжения (кПа):\n{np.round(eigenvalues, 1)}")

            # Вывод шаровой и девиаторной части главных напряжений
            print_matrix(sigma_spherical_principal, "Шаровая часть главных напряжений (кПа)", 1)
            print_matrix(sigma_deviatoric_principal, "Девиаторная часть главных напряжений (кПа)", 1)

            # Вывод матриц деформаций с точностью до 5 знаков после запятой
            print_matrix(epsilon_spherical, "Шаровая часть деформаций", 5)
            print_matrix(epsilon_deviatoric, "Девиаторная часть деформаций", 5)
            print_matrix(epsilon, "Полный тензор деформаций", 5)

            if strength_check:
                st.write("\nПрочность обеспечена (точка внутри пирамиды прочности).")
            else:
                st.write("\nПрочность не обеспечена (точка на грани или за пределами пирамиды прочности).")




        # Обновление графика
        update_plot(eigenvalues, c, phi)  # Исправленный вызов

    except ValueError:
        st.error("Проверьте введенные данные!")

def check_mohr_coulomb_strength(sigma1, sigma2, sigma3, c, phi):
    def strength_condition(sigma_i, sigma_j):
        return abs(sigma_i - sigma_j) <= 2 * c * np.cos(phi) / (1 - np.sin(phi))

    return (strength_condition(sigma1, sigma2) and
            strength_condition(sigma1, sigma3) and
            strength_condition(sigma2, sigma3))

# Интерфейс

# Ввод данных
st.sidebar.header("Тензор напряжений")
sigma11 = st.sidebar.number_input("σ₁₁", value=30.0, key="sigma11", on_change=calculate)
tau12 = st.sidebar.number_input("τ₁₂", value=10.0, key="tau12", on_change=calculate)
tau13 = st.sidebar.number_input("τ₁₃", value=0.0, key="tau13", on_change=calculate)
sigma22 = st.sidebar.number_input("σ₂₂", value=30.0, key="sigma22", on_change=calculate)
tau23 = st.sidebar.number_input("τ₂₃", value=0.0, key="tau23", on_change=calculate)
sigma33 = st.sidebar.number_input("σ₃₃", value=30.0, key="sigma33", on_change=calculate)

st.sidebar.header("Механические характеристики")
E = st.sidebar.number_input("Модуль Юнга E", value=10000.0, key="E", on_change=calculate)
nu = st.sidebar.number_input("Poisson's ratio ν", value=0.3, key="nu", on_change=calculate)
c = st.sidebar.number_input("Сцепление c", value=10.0, key="c", on_change=calculate)
phi = st.sidebar.number_input("Угол трения φ (градусы)", value=20.0, key="phi", on_change=calculate)

# Выполнение расчета при загрузке
if 'init' not in st.session_state:
    st.session_state.init = True
    calculate()
