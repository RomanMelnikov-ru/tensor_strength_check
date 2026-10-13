import numpy as np
import streamlit as st
import plotly.graph_objs as go
from math import radians

# ============================================================
# ЦВЕТОВАЯ СХЕМА ДЛЯ МЕРИДИАНОВ
# ============================================================
COLOR_TC = '#FFA500'  # Оранжевый - Трехосное сжатие (Triaxial Compression)
COLOR_TE = '#FF4500'  # Красно-оранжевый - Трехосное растяжение (Triaxial Extension)

st.set_page_config(
    layout="wide",
    page_title="Анализ НДС",
    initial_sidebar_state="expanded"
)

# ============================================================
# БЛОК 1. ФУНКЦИИ РАСЧЁТА
# ============================================================

def calculate_moduli(E, nu):
    """Вычисляет модуль сдвига G и объёмный модуль K."""
    G = E / (2 * (1 + nu))
    K = E / (3 * (1 - 2 * nu))
    return G, K


def split_stress_tensor(sigma):
    """Разложение тензора напряжений на шаровую и девиаторную части."""
    sigma_mean = np.trace(sigma) / 3
    sigma_spherical = sigma_mean * np.eye(3)
    sigma_deviatoric = sigma - sigma_spherical
    return sigma_spherical, sigma_deviatoric


def calculate_strain_tensor(sigma, G, K):
    """Вычисляет тензор деформаций по закону Гука."""
    sigma_spherical, sigma_deviatoric = split_stress_tensor(sigma)
    epsilon_spherical = sigma_spherical / (3 * K)
    epsilon_deviatoric = sigma_deviatoric / (2 * G)
    epsilon = epsilon_spherical + epsilon_deviatoric
    return epsilon_spherical, epsilon_deviatoric, epsilon


def calculate_principal_stresses(sigma):
    """Главные напряжения и их направления (по убыванию)."""
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors


def calculate_spherical_and_deviatoric_principal_stresses(eigenvalues):
    """Шаровая и девиаторная части главных напряжений."""
    eigenvalues = np.real(eigenvalues).astype(float)
    sigma_mean = float(np.mean(eigenvalues))
    sigma_spherical = sigma_mean * np.eye(3)
    sigma_deviatoric = np.diag(eigenvalues) - sigma_spherical
    return sigma_spherical, sigma_deviatoric


def check_mohr_coulomb_strength(sigma1, sigma2, sigma3, c, phi):
    """Проверка прочности по критерию Мора–Кулона."""
    def strength_condition(si, sj):
        return abs(si - sj) <= 2 * c * np.cos(phi) / (1 - np.sin(phi))
    return (strength_condition(sigma1, sigma2) and
            strength_condition(sigma1, sigma3) and
            strength_condition(sigma2, sigma3))


def stress_invariants(sigma):
    """Вычисляет инварианты напряжений: I1, J2, J3, p, q, угол Лоде."""
    sigma = np.real(sigma).astype(float)

    # Первый инвариант тензора напряжений
    I1 = float(np.trace(sigma))

    # Среднее напряжение
    p = I1 / 3.0

    # Девиатор напряжений
    s = sigma - p * np.eye(3)

    # Второй и третий инварианты девиатора
    J2 = 0.5 * float(np.sum(s * s))          # s : s
    J3 = float(np.trace(s @ s @ s)) / 3.0    # (1/3) tr(s^3)

    # Девиаторное напряжение
    q = float(np.sqrt(3.0 * J2))

    # Угол Лоде – Надаи
    if J2 < 1e-12:
        theta = 0.0
    else:
        arg = -1.5 * np.sqrt(3.0) * J3 / (J2 ** 1.5)
        arg = np.clip(arg, -1.0, 1.0)
        theta = np.arcsin(arg) / 3.0

    return {
        "I1": I1,
        "J2": J2,
        "J3": J3,
        "p": p,
        "q": q,
        "lode_rad": float(theta),
        "lode_deg": float(np.degrees(theta)),
    }


def strength_envelope_pq(p_array, c, phi):
    """
    Границы Мора-Кулона в плоскости p–q.
    Возвращает два массива q для трехосного сжатия и растяжения.
    """
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    # Трехосное сжатие (Triaxial Compression, theta = +30°)
    M_tc = 6.0 * sin_phi / (3.0 - sin_phi)
    k_tc = 6.0 * c * cos_phi / (3.0 - sin_phi)
    q_tc = M_tc * p_array + k_tc

    # Трехосное растяжение (Triaxial Extension, theta = -30°)
    # В критерии Мора-Кулона обе линии выходят из одной точки на оси p.
    # Формула для наклона TE: M_te = 6*sin(phi) / (3 + sin(phi))
    # Отсечка k_te должна быть такой, чтобы линия прошла через ту же вершину,
    # что и TC. Вершина пирамиды находится при p = -c / tan(phi).
    # Но для 2D графика проще использовать прямые формулы:
    M_te = 6.0 * sin_phi / (3.0 + sin_phi)
    k_te = 6.0 * c * cos_phi / (3.0 + sin_phi)
    q_te = M_te * p_array + k_te

    return q_tc, q_te


def meridian_traces_in_3d(s1, s2, s3, c, phi):
    """
    Возвращает 2 луча из вершины пирамиды, лежащие в меридианной
    плоскости (проходящей через гидроось и точку (s1,s2,s3)),
    вдоль граней MC-пирамиды.

    Луч: (V, direction) — от вершины пирамиды в направлении direction.
    """
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    V_val = -c / np.tan(phi)
    V = np.array([V_val, V_val, V_val])

    # Нормаль меридианной плоскости
    point = np.array([s1, s2, s3], dtype=float)
    hydro = np.array([1.0, 1.0, 1.0])
    n_mer = np.cross(hydro, point)
    n_mer /= np.linalg.norm(n_mer)

    # 6 нормалей граней пирамиды MC (в системе σ1, σ2, σ3).
    # Каждая грань: n · X = d, где n — внешняя нормаль.
    # Грани соответствуют активным парам главных напряжений.
    # Используем пары (i, j): (1,3), (3,1), (1,2), (2,1), (2,3), (3,2).
    # sigma_i - sigma_j = 2c·cosφ - (sigma_i + sigma_j)·sinφ
    # → sigma_i(1+sinφ) - sigma_j(1-sinφ) = 2c·cosφ
    # n = (1+sinφ)·e_i - (1-sinφ)·e_j,  d = 2c·cosφ

    faces = []
    for (i, j) in [(0, 2), (2, 0), (0, 1), (1, 0), (1, 2), (2, 1)]:
        n = np.zeros(3)
        n[i] = 1.0 + sin_phi
        n[j] = -(1.0 - sin_phi)
        faces.append((n, 2.0 * c * cos_phi))

    # Направления пересечения меридианной плоскости с каждой гранью
    directions = []
    for n_face, d_face in faces:
        d_line = np.cross(n_mer, n_face)
        norm_d = np.linalg.norm(d_line)
        if norm_d < 1e-9:
            continue
        d_line /= norm_d

        # Ориентация: выбираем то направление, которое «смотрит в сторону
        # положительных напряжений» (чтобы луч шёл из вершины в тело пирамиды).
        # Проверка: смещаемся вдоль d_line на малое ε и смотрим, растёт ли
        # расстояние от точки до гидрооси (т.е. уходим от вершины).
        eps = 1.0
        pt_test = V + eps * d_line
        # Если y-координата (σ1) меньше, чем у V — переворачиваем знак
        if pt_test[0] < V[0]:
            d_line = -d_line
        directions.append(d_line)

    # Отбор 2 «внешних» направлений. Используем угол в π-плоскости:
    # проекция каждого направления на π-плоскость должна быть
    # максимально «в сторону» точки НДС, но так, чтобы направление лежало
    # в меридианной плоскости (по построению).
    #
    # Практический критерий: из 6 направлений оставляем те 2, что дают
    # проекцию на π-плоскость, наиболее близкую по углу к направлению точки.

    # Направление точки в π-плоскости
    x_pt = (s3 - s1) / np.sqrt(2.0)
    y_pt = (2.0 * s2 - s1 - s3) / np.sqrt(6.0)
    ang_pt = np.arctan2(y_pt, x_pt)

    def proj_angle(d):
        # Проекция направления на π-плоскость
        # d — вектор в (σ1, σ2, σ3); проекция: x = (d3 - d1)/√2, y = (2d2 - d1 - d3)/√6
        x = (d[2] - d[0]) / np.sqrt(2.0)
        y = (2.0 * d[1] - d[0] - d[2]) / np.sqrt(6.0)
        return np.arctan2(y, x)

    angs = [proj_angle(d) for d in directions]

    # Угловое расстояние до ang_pt (по модулю, с учётом цикличности)
    def ang_diff(a, b):
        d = abs(a - b) % (2 * np.pi)
        return min(d, 2 * np.pi - d)

    # Сортируем по угловой близости и берём 2 ближайших
    order = sorted(range(len(directions)),
                   key=lambda k: ang_diff(angs[k], ang_pt))
    chosen = [directions[order[0]], directions[order[1]]]
    return V, chosen


# ============================================================
# БЛОК 2. ГРАНИЦЫ ПИРАМИДЫ МОРА-КУЛОНА (без изменений)
# ============================================================

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

# ============================================================
# БЛОК 2б. СЕЧЕНИЕ ПИРАМИДЫ МОРА–КУЛОНА ПЛОСКОСТЬЮ
# ============================================================

def pyramid_vertices(c, phi, p_max):
    """
    Возвращает 7 вершин MC-пирамиды:
      [V, TC1, TE2, TC3, TE1, TC2, TE3]  — в порядке обхода шестиугольника.
    V = (-c/tanφ, -c/tanφ, -c/tanφ) — вершина пирамиды на гидрооси.
    Остальные 6 — на девиаторной плоскости p = p_max.
    """
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    V_val = -c / np.tan(phi)
    vertex = np.array([V_val, V_val, V_val])

    d_TC = (6.0 * sin_phi * p_max + 6.0 * c * cos_phi) / (3.0 - sin_phi)
    d_TE = (6.0 * sin_phi * p_max + 6.0 * c * cos_phi) / (3.0 + sin_phi)

    tc_eq  = p_max - d_TC / 3.0
    tc_big = p_max + 2.0 * d_TC / 3.0
    te_eq    = p_max + d_TE / 3.0
    te_small = p_max - 2.0 * d_TE / 3.0

    hex_pts = [
        np.array([tc_big,   tc_eq,    tc_eq]),      # TC1
        np.array([te_eq,    te_small, te_eq]),      # TE2
        np.array([tc_eq,    tc_eq,    tc_big]),     # TC3
        np.array([te_small, te_eq,    te_eq]),      # TE1
        np.array([tc_eq,    tc_big,   tc_eq]),      # TC2
        np.array([te_eq,    te_eq,    te_small]),   # TE3
    ]
    return [vertex] + hex_pts


def pyramid_edges(c, phi, p_max):
    """
    Возвращает список рёбер пирамиды MC как пар точек (p1, p2).
    Рёбра: V — каждая из 6 вершин шестиугольника, плюс 6 рёбер шестиугольника.
    """
    pts = pyramid_vertices(c, phi, p_max)
    V = pts[0]
    hex6 = pts[1:]

    edges = []
    # 6 рёбер из вершины в каждую вершину шестиугольника
    for h in hex6:
        edges.append((V, h))
    # 6 рёбер вдоль шестиугольника (замыкаем)
    for i in range(6):
        edges.append((hex6[i], hex6[(i + 1) % 6]))
    return edges


def slice_edges_with_plane(edges, origin, normal, tol=1e-9):
    """
    Пересечение списка рёбер (пар точек) с плоскостью (origin, normal).
    Возвращает список 3D-точек пересечения.
    """
    n = np.asarray(normal, dtype=float)
    n /= np.linalg.norm(n)
    o = np.asarray(origin, dtype=float)

    intersections = []
    for (p1, p2) in edges:
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        d1 = float(np.dot(p1 - o, n))
        d2 = float(np.dot(p2 - o, n))
        if abs(d1 - d2) < tol:
            continue
        if d1 * d2 <= 0:
            t = d1 / (d1 - d2)
            pt = p1 + t * (p2 - p1)
            intersections.append(pt)

    # Уникализация
    unique = []
    for pt in intersections:
        is_new = True
        for q in unique:
            if np.linalg.norm(pt - q) < 1e-6:
                is_new = False
                break
        if is_new:
            unique.append(pt)
    return unique

# ============================================================
# БЛОК 3. ГРАФИКИ
# ============================================================

def update_plot(eigenvalues, c, phi, show_pyramid,
                show_deviator_plane=False, show_meridian_plane=False):
    """3D-график НДС, пирамиды Мора–Кулона и опциональных плоскостей."""
    fig = go.Figure()

    s1, s2, s3 = float(eigenvalues[0]), float(eigenvalues[1]), float(eigenvalues[2])
    p_mean_pt = (s1 + s2 + s3) / 3.0

    # Гидростатическая ось
    hydrostatic_axis = np.linspace(-10, 50, 50).tolist()
    fig.add_trace(go.Scatter3d(
        x=hydrostatic_axis, y=hydrostatic_axis, z=hydrostatic_axis,
        mode='lines', line=dict(color='black', dash='dash'),
        name="Гидростатическая ось"
    ))

    # --------- Оси главных напряжений со стрелками ---------
    # Вспомогательная функция для отрисовки оси со стрелкой
    def add_arrow_axis(fig, direction, color, label, length=60.0):
        """
        Рисует ось в заданном направлении с конусом-стрелкой на конце.
        direction: np.array([dx, dy, dz]) — единичный вектор направления
        color: цвет оси
        label: подпись (например, "σ₁")
        length: длина оси
        """
        # Начало и конец линии
        start = np.array([0.0, 0.0, 0.0])
        end = direction * length

        # 1. Линия оси
        fig.add_trace(go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode='lines',
            line=dict(color=color, width=4),
            name=label,
            showlegend=True,
        ))

        # 2. Конус-стрелка на конце
        # Plotly Cone принимает координаты НАЧАЛА вектора (x,y,z)
        # и компоненты вектора (u,v,w). Конус рисуется от точки (x,y,z)
        # в направлении (u,v,w).
        # Чтобы стрелка «сидела» на конце оси, сместим её немного назад.
        arrow_length = 0.08 * length  # длина стрелки
        arrow_pos = end - direction * arrow_length  # начало конуса
        fig.add_trace(go.Cone(
            x=[arrow_pos[0]],
            y=[arrow_pos[1]],
            z=[arrow_pos[2]],
            u=[direction[0] * arrow_length],
            v=[direction[1] * arrow_length],
            w=[direction[2] * arrow_length],
            colorscale=[[0, color], [1, color]],
            showscale=False,
            sizemode='absolute',
            sizeref=arrow_length,
            anchor='tail',
            showlegend=False,
            hoverinfo='skip',
        ))

    # Направления осей (в системе координат графика: x=σ₂, y=σ₃, z=σ₁)
    # Ось σ₁ — вертикально вверх (z)
    add_arrow_axis(fig, np.array([0.0, 0.0, 1.0]), 'red', 'σ₁', length=60.0)
    # Ось σ₂ — вдоль x
    add_arrow_axis(fig, np.array([1.0, 0.0, 0.0]), 'green', 'σ₂', length=60.0)
    # Ось σ₃ — вдоль y
    add_arrow_axis(fig, np.array([0.0, 1.0, 0.0]), 'blue', 'σ₃', length=60.0)

    # Точка НДС
    fig.add_trace(go.Scatter3d(
        x=[s2], y=[s3], z=[s1], mode='markers',
        marker=dict(color='red', size=5), name="Напряженное состояние"
    ))

    # Разложение на шаровую и девиаторную части
    hydro_mean = float(np.mean([s1, s2, s3]))
    fig.add_trace(go.Scatter3d(
        x=[0, hydro_mean], y=[0, hydro_mean], z=[0, hydro_mean],
        mode='lines', line=dict(color='magenta', width=3),
        name="Шаровая часть напряжения"
    ))
    fig.add_trace(go.Scatter3d(
        x=[hydro_mean, s2], y=[hydro_mean, s3], z=[hydro_mean, s1],
        mode='lines', line=dict(color='green', width=3),
        name="Девиаторная часть напряжения"
    ))

    # --------- Пирамида Мора–Кулона (без изменений) ---------
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

    # --------- След пирамиды на девиаторной и меридианной плоскостях ---------
    if show_pyramid and (show_deviator_plane or show_meridian_plane):
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        # ---------------------------------------------------------
        # 1. След на девиаторной плоскости (шестиугольник при p = σm)
        # ---------------------------------------------------------
        if show_deviator_plane:
            p_pt = float(np.mean([s1, s2, s3]))
            d_TC = (6.0 * sin_phi * p_pt + 6.0 * c * cos_phi) / (3.0 - sin_phi)
            d_TE = (6.0 * sin_phi * p_pt + 6.0 * c * cos_phi) / (3.0 + sin_phi)

            tc_eq  = p_pt - d_TC / 3.0
            tc_big = p_pt + 2.0 * d_TC / 3.0
            te_eq    = p_pt + d_TE / 3.0
            te_small = p_pt - 2.0 * d_TE / 3.0

            # 6 вершин в порядке обхода
            hex_sigma = [
                (tc_big,   tc_eq,    tc_eq),      # TC1
                (te_eq,    te_small, te_eq),      # TE2
                (tc_eq,    tc_eq,    tc_big),     # TC3
                (te_small, te_eq,    te_eq),      # TE1
                (tc_eq,    tc_big,   tc_eq),      # TC2
                (te_eq,    te_eq,    te_small),   # TE3
            ]
            # Замыкаем
            hex_sigma_closed = hex_sigma + [hex_sigma[0]]

            # Перестановка (σ1, σ2, σ3) -> (x=σ2, y=σ3, z=σ1)
            hx = [float(p[1]) for p in hex_sigma_closed]
            hy = [float(p[2]) for p in hex_sigma_closed]
            hz = [float(p[0]) for p in hex_sigma_closed]

            fig.add_trace(go.Scatter3d(
                x=hx, y=hy, z=hz,
                mode='lines',
                line=dict(color='crimson', width=5),
                name='След пирамиды на π-плоскости',
                legendgroup='trace_on_planes',
                showlegend=True,
            ))

        # ---------------------------------------------------------
        # 2. След на меридианной плоскости — честное сечение
        # ---------------------------------------------------------
        if show_meridian_plane:
            point = np.array([s1, s2, s3], dtype=float)
            hydro = np.array([1.0, 1.0, 1.0])
            n_mer_raw = np.cross(hydro, point)
            nv = np.linalg.norm(n_mer_raw)
            if nv > 1e-9:
                n_mer = n_mer_raw / nv

                # Уровень обрезки пирамиды
                p_max = 150.0 / 3
                edges = pyramid_edges(c, phi, p_max)
                origin = np.zeros(3)
                pts_in_plane = slice_edges_with_plane(edges, origin, n_mer)

                if len(pts_in_plane) >= 2:
                    # Сортируем по p (среднему напряжению)
                    pts_sorted = sorted(pts_in_plane,
                                        key=lambda p: (p[0] + p[1] + p[2]) / 3.0)
                    V_pt = pts_sorted[0]   # вершина
                    far_pts = pts_sorted[1:]  # дальние точки

                    # Определяем для каждой дальней точки, TC это или TE.
                    # Критерий: в направлении TC напряжение σ1 растёт быстрее,
                    # чем σ3. Сравним приращения (far - V).
                    # Для TC: Δσ1 > Δσ3 (луч идёт в сторону увеличения σ1)
                    # Для TE: Δσ3 > Δσ1
                    for idx, far in enumerate(far_pts):
                        delta = far - V_pt
                        # delta[0] = Δσ1, delta[2] = Δσ3
                        # В системе (σ1, σ2, σ3) ось z = σ1, x = σ2, y = σ3
                        # Но в plotly мы переставляем: x=σ2, y=σ3, z=σ1
                        # Поэтому delta в исходных координатах: (Δσ1, Δσ2, Δσ3)
                        # Δσ1 = delta[0], Δσ3 = delta[2]
                        if delta[0] > delta[2]:
                            # TC: сжатие
                            line_color = COLOR_TC
                            line_name = 'Меридиан сжатия (TC)'
                        else:
                            # TE: растяжение
                            line_color = COLOR_TE
                            line_name = 'Меридиан растяжения (TE)'

                        # Координаты для plotly: x=σ2, y=σ3, z=σ1
                        xs = [float(V_pt[1]), float(far[1])]
                        ys = [float(V_pt[2]), float(far[2])]
                        zs = [float(V_pt[0]), float(far[0])]

                        fig.add_trace(go.Scatter3d(
                            x=xs, y=ys, z=zs,
                            mode='lines',
                            line=dict(color=line_color, width=5),  # Толще
                            name=line_name,
                            legendgroup='trace_on_planes',
                            showlegend=True,
                        ))

    # --------------------------------------------------------------
    # Девиаторная (π-) плоскость: σ₁ + σ₂ + σ₃ = 3·σm
    # --------------------------------------------------------------
    if show_deviator_plane:
        sigma_m = float(np.mean([s1, s2, s3]))
        u = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
        v = np.array([1.0, 1.0, -2.0]) / np.sqrt(6.0)
        center = np.array([sigma_m, sigma_m, sigma_m])

        L = 90.0
        corners_uv = [(-L, -L), (+L, -L), (+L, +L), (-L, +L)]
        pts = [center + cu * u + cv * v for cu, cv in corners_uv]
        # ВАЖНО: переставляем (σ1, σ2, σ3) -> (x=σ2, y=σ3, z=σ1)
        xs = [float(p[1]) for p in pts]
        ys = [float(p[2]) for p in pts]
        zs = [float(p[0]) for p in pts]

        fig.add_trace(go.Mesh3d(
            x=xs, y=ys, z=zs,
            i=[0, 0], j=[1, 2], k=[2, 3],
            color='deepskyblue', opacity=0.10,
            name='Девиаторная (π-) плоскость',
            showlegend=True, legendgroup='dev_plane',
            hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter3d(
            x=xs + [xs[0]], y=ys + [ys[0]], z=zs + [zs[0]],
            mode='lines', line=dict(color='deepskyblue', width=2),
            name='Контур π-плоскости',
            legendgroup='dev_plane', showlegend=False, hoverinfo='skip',
        ))

    # --------------------------------------------------------------
    # Меридианная плоскость: содержит гидроось (1,1,1) и точку (σ1,σ2,σ3)
    # --------------------------------------------------------------
    if show_meridian_plane:
        point = np.array([s1, s2, s3], dtype=float)
        hydro = np.array([1.0, 1.0, 1.0])

        normal = np.cross(hydro, point)
        norm_val = np.linalg.norm(normal)
        if norm_val < 1e-9:
            st.info("Меридианная плоскость не определена: точка лежит на гидростатической оси.")
        else:
            normal /= norm_val
            e1 = hydro / np.linalg.norm(hydro)  # вдоль гидрооси
            proj = point - np.dot(point, e1) * e1  # проекция P ⟂ e1
            e2 = proj / np.linalg.norm(proj)  # нормированный ⟂ e1

            # Точка на плоскости — начало координат (гидроось проходит через O)
            origin = np.zeros(3)

            M = 100.0  # вдоль гидрооси
            L = 100.0  # вдоль e2 (девиаторное направление)

            # 4 угла параллелограмма как СМЕЩЕНИЯ от origin
            pts = [
                origin - M * e1 - L * e2,
                origin - M * e1 + L * e2,
                origin + M * e1 + L * e2,
                origin + M * e1 - L * e2,
            ]
            # Перестановка (σ1, σ2, σ3) -> (x=σ2, y=σ3, z=σ1)
            xs = [float(p[1]) for p in pts]
            ys = [float(p[2]) for p in pts]
            zs = [float(p[0]) for p in pts]

            fig.add_trace(go.Mesh3d(
                x=xs, y=ys, z=zs,
                i=[0, 0], j=[1, 2], k=[2, 3],
                color='mediumseagreen', opacity=0.10,
                name='Меридианная плоскость',
                showlegend=True, legendgroup='mer_plane',
                hoverinfo='skip',
            ))
            fig.add_trace(go.Scatter3d(
                x=xs + [xs[0]], y=ys + [ys[0]], z=zs + [zs[0]],
                mode='lines', line=dict(color='mediumseagreen', width=2),
                name='Контур меридианной плоскости',
                legendgroup='mer_plane', showlegend=False, hoverinfo='skip',
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


def update_pq_plot(inv, c, phi, sigma1, sigma3):
    """2D-график в плоскости p–q."""
    fig = go.Figure()

    # Вершина пирамиды на оси p
    p_vertex = -c / np.tan(phi) if np.tan(phi) != 0 else 0.0

    p_min = min(p_vertex - 10.0, inv["p"] - 30.0)
    p_max = max(200.0, inv["p"] + 50.0)
    p_arr = np.linspace(p_min, p_max, 400)

    q_tc, q_te = strength_envelope_pq(p_arr, c, phi)

    # Верхняя граница (triaxial compression) - сплошная, цвет TC
    fig.add_trace(go.Scatter(
        x=p_arr, y=q_tc,
        mode='lines',
        line=dict(color=COLOR_TC, width=3), # Сплошная линия
        name='Мора–Кулон (сжатие, θ=+30°)',
    ))

    # Нижняя граница (triaxial extension) - сплошная, цвет TE
    fig.add_trace(go.Scatter(
        x=p_arr, y=q_te,
        mode='lines',
        line=dict(color=COLOR_TE, width=3), # Убрали dash='dash', сделали сплошной
        name='Мора–Кулон (растяжение, θ=−30°)',
    ))

    # Линия гидростатической оси q = 0
    fig.add_trace(go.Scatter(
        x=[p_min, p_max], y=[0, 0],
        mode='lines',
        line=dict(color='black', width=1, dash='dot'),
        name='q = 0 (гидростатическая ось)',
    ))

    # Точка напряжённого состояния
    fig.add_trace(go.Scatter(
        x=[inv["p"]], y=[inv["q"]],
        mode='markers+text',
        marker=dict(color='red', size=12, symbol='circle',
                    line=dict(color='darkred', width=2)),
        text=[f"  p={inv['p']:.1f}, q={inv['q']:.1f}"],
        textposition='top right',
        name='Напряжённое состояние',
    ))

    # Отрезок от точки до оси q=0 (наглядность)
    fig.add_trace(go.Scatter(
        x=[inv["p"], inv["p"]], y=[0, inv["q"]],
        mode='lines',
        line=dict(color='red', width=1, dash='dot'),
        showlegend=False,
    ))

    # Вертикальная линия в вершине пирамиды
    fig.add_vline(x=p_vertex, line_dash="dash", line_color="gray", opacity=0.5,
                  annotation_text=f"Вершина p={p_vertex:.1f}",
                  annotation_position="bottom right")

    fig.update_layout(
        title=f"Плоскость p–q   (θ = {inv['lode_deg']:.1f}°)",
        xaxis_title="p = (σ₁+σ₂+σ₃)/3, кПа",
        yaxis_title="q = √(3J₂), кПа",
        height=600,
        margin=dict(l=60, r=20, t=50, b=50),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        uirevision="pq_constant",
        xaxis=dict(range=[p_min, p_max]),
        yaxis=dict(range=[0, max(np.max(q_tc), inv["q"]) * 1.2])
    )

    st.plotly_chart(fig, use_container_width=True, config={
        'displayModeBar': True,
        'responsive': True,
    })


def update_deviator_plot(inv, c, phi, eigenvalues):
    """
    2D-график в девиаторной (π-) плоскости.
    Ориентация осей согласована с 3D-графиком:
      σ₁ — вверх (+90°)
      σ₂ — влево-вниз (+210° = −150°)
      σ₃ — вправо-вниз (−30°)
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    p = inv["p"]

    d_TC = (6.0 * sin_phi * p + 6.0 * c * cos_phi) / (3.0 - sin_phi)
    d_TE = (6.0 * sin_phi * p + 6.0 * c * cos_phi) / (3.0 + sin_phi)

    tc_eq  = p - d_TC / 3.0
    tc_big = p + 2.0 * d_TC / 3.0
    te_eq    = p + d_TE / 3.0
    te_small = p - 2.0 * d_TE / 3.0

    # Координаты в π-плоскости (канонические, θ = +30° для σ₁)
    def to_xy(s1, s2, s3):
        x = (s3 - s2) / np.sqrt(2.0)
        y = (2.0 * s1 - s2 - s3) / np.sqrt(6.0)
        return x, y

    # --- 6 вершин шестиугольника (обход по кругу) ---
    # В канонических координатах (x = (σ₃−σ₂)/√2, y = (2σ₁−σ₂−σ₃)/√6)
    # вершина TCσ₁ лежит при θ=+30°, TEσ₂ — при θ=+150°, TCσ₂ — при θ=−90° и т.д.
    hex_vertices_named = [
        ("TCσ₁", to_xy(tc_big,   tc_eq,    tc_eq)),
        ("TEσ₂", to_xy(te_eq,    te_small, te_eq)),
        ("TCσ₃", to_xy(tc_eq,    tc_eq,    tc_big)),
        ("TEσ₁", to_xy(te_small, te_eq,    te_eq)),
        ("TCσ₂", to_xy(tc_eq,    tc_big,   tc_eq)),
        ("TEσ₃", to_xy(te_eq,    te_eq,    te_small)),
    ]
    hex_xy_closed = [pt for _, pt in hex_vertices_named] + [hex_vertices_named[0][1]]

    # --- Рабочая область: четырёхугольник (0,0) → TCσ₁ → TEσ₃ → TCσ₂ → (0,0) ---
    # Это пересечение фундаментального сектора (между осями σ₁ и σ₂)
    # с шестиугольником Мора–Кулона.
    pt_TC1 = hex_vertices_named[0][1]  # TCσ₁  (угол +90°)
    pt_TE3 = hex_vertices_named[5][1]  # TEσ₃  (угол +150°)
    pt_TC2 = hex_vertices_named[4][1]  # TCσ₂  (угол -150° = +210°)

    work_x = [0.0, pt_TC1[0], pt_TE3[0], pt_TC2[0], 0.0]
    work_y = [0.0, pt_TC1[1], pt_TE3[1], pt_TC2[1], 0.0]

    # Полупрозрачная заливка
    fig.add_trace(go.Scatter(
        x=work_x, y=work_y,
        fill='toself',
        fillcolor='rgba(255, 165, 0, 0.25)',
        line=dict(color='rgba(255, 165, 0, 0.0)'),
        name='Рабочая область (фундаментальный сектор)',
        hoverinfo='skip',
        showlegend=True,
        legendgroup='working_area',
    ))

    # Штриховка рабочей области
    try:
        fig.add_trace(go.Scatter(
            x=work_x, y=work_y,
            fill='toself',
            fillcolor='rgba(255, 165, 0, 0.0)',
            line=dict(color='rgba(255, 165, 0, 0.0)'),
            fillpattern=dict(
                shape='/',
                fgcolor='rgba(200, 120, 0, 0.6)',
                bgcolor='rgba(255, 255, 255, 0.0)',
                fillmode='overlay',
                size=8,
                solidity=0.4,
            ),
            name='Штриховка',
            hoverinfo='skip',
            showlegend=False,
            legendgroup='working_area',
        ))
    except Exception:
        pass

    # --- Контур шестиугольника ---
    fig.add_trace(go.Scatter(
        x=[pt[0] for pt in hex_xy_closed],
        y=[pt[1] for pt in hex_xy_closed],
        mode='lines+markers',
        line=dict(color='crimson', width=3),
        marker=dict(color='crimson', size=8),
        name='След пирамиды Мора–Кулона',
    ))

    # --- Оси со стрелками ---
    # ВАЖНО: углы подобраны под ориентацию 3D-графика
    # σ₁ — вверх (+90°)
    # σ₂ — влево-вниз (210° = −150°)
    # σ₃ — вправо-вниз (−30°)
    max_r = max(np.hypot(x, y) for _, pt in hex_vertices_named for x, y in [pt])
    axis_len = max_r * 1.25

    axes_def = [
        ("σ₁",  np.pi / 2.0,           "red"),    # +90°
        ("σ₂",  7.0 * np.pi / 6.0,     "green"),  # +210° = −150°
        ("σ₃", -np.pi / 6.0,           "blue"),   # −30°
    ]

    for name, ang_axis, color in axes_def:
        x1 = axis_len * np.cos(ang_axis)
        y1 = axis_len * np.sin(ang_axis)

        fig.add_trace(go.Scatter(
            x=[0.0, x1], y=[0.0, y1],
            mode='lines',
            line=dict(color=color, width=2, dash='dot'),
            name=f"Ось {name}",
            showlegend=True,
            legendgroup='axes',
        ))

        # Стрелка
        fig.add_annotation(
            x=x1, y=y1,
            ax=0.0, ay=0.0,
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor=color,
            text='',
        )

        # Подпись
        label_r = axis_len * 1.12
        fig.add_annotation(
            x=label_r * np.cos(ang_axis),
            y=label_r * np.sin(ang_axis),
            text=f"<b>{name}</b>",
            showarrow=False,
            font=dict(color=color, size=14),
            xref='x', yref='y',
        )

    # --- Гидростатическая ось ---
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode='markers',
        marker=dict(color='black', size=10, symbol='x'),
        name='Гидростатическая ось',
    ))

    # --- Точка НДС ---
    s1, s2, s3 = float(eigenvalues[0]), float(eigenvalues[1]), float(eigenvalues[2])
    x_pt, y_pt = to_xy(s1, s2, s3)

    fig.add_trace(go.Scatter(
        x=[x_pt], y=[y_pt],
        mode='markers+text',
        marker=dict(color='red', size=14, symbol='circle',
                    line=dict(color='darkred', width=2)),
        text=[f"  θ={inv['lode_deg']:.1f}°, ρ={np.hypot(x_pt, y_pt):.1f}"],
        textposition='top right',
        name='Напряжённое состояние',
    ))

    fig.add_trace(go.Scatter(
        x=[0, x_pt], y=[0, y_pt], mode='lines',
        line=dict(color='red', width=1, dash='dot'),
        showlegend=False,
    ))

    # --- Оформление ---
    fig.update_layout(
        title=f"След пирамиды Мора–Кулона в π-плоскости при p = {p:.1f} кПа",
        xaxis_title="x",
        yaxis_title="y",
        height=700,
        margin=dict(l=60, r=20, t=60, b=50),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        uirevision="dev_constant",
    )

    st.plotly_chart(fig, use_container_width=True, config={
        'displayModeBar': True,
        'responsive': True,
    })


# ============================================================
# БЛОК 4. ТЕОРИЯ (содержимое гармошек)
# ============================================================

FORMULAS_MAIN = r"""
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

### 5. Инварианты напряжений и параметры p–q

$$
p = \frac{I_1}{3} = \frac{\sigma_1 + \sigma_2 + \sigma_3}{3}
$$

$$
J_2 = \frac{1}{2}\,s_{ij}s_{ij}, \qquad
J_3 = \frac{1}{3}\,s_{ij}s_{jk}s_{ki}, \qquad
s_{ij} = \sigma_{ij} - p\,\delta_{ij}
$$

$$
q = \sqrt{3 J_2}
$$

### 6. Угол Лоде – Надаи

$$
\theta = \frac{1}{3}\arcsin\!\left(-\frac{3\sqrt{3}}{2}\cdot\frac{J_3}{J_2^{3/2}}\right),
\qquad -\frac{\pi}{6} \le \theta \le \frac{\pi}{6}
$$

или через главные напряжения:

$$
\theta = \arctan\!\left(\frac{1}{\sqrt{3}}\cdot\frac{2\sigma_2 - \sigma_1 - \sigma_3}{\sigma_1 - \sigma_3}\right)
$$

Диапазоны:
- θ = −30° — трёхосное растяжение,
- θ = 0° — чистый сдвиг,
- θ = +30° — трёхосное сжатие.

### 7. Поверхность Мора – Кулона в плоскости p–q

$$
q = M(\theta)\,p + k(\theta),
\qquad
M(\theta) = \frac{6\sin\varphi}{3-\sin\varphi\sin 3\theta},
\qquad
k(\theta) = \frac{6c\cos\varphi}{3-\sin\varphi\sin 3\theta}
$$

Предельные случаи:

$$
q_{tc} = \frac{6\sin\varphi}{3-\sin\varphi}\,p + \frac{6c\cos\varphi}{3-\sin\varphi}
\quad(\theta = +30°)
$$

$$
q_{te} = \frac{6\sin\varphi}{3+\sin\varphi}\,p + \frac{6c\cos\varphi}{3+\sin\varphi}
\quad(\theta = -30°)
$$

и аналогичными соотношениями для остальных пяти рёбер, полученных циклической перестановкой индексов.
"""

FORMULAS_DEVIATOR = r"""
### Девиаторная (π-) плоскость

Девиаторная плоскость — это плоскость, перпендикулярная гидростатической оси
$\sigma_1 = \sigma_2 = \sigma_3$ и проходящая через начало координат
(или через точку с заданным средним напряжением $p$).

### 1. Координаты в π-плоскости

Проекция тензора напряжений на девиаторную плоскость описывается двумя
координатами:

$$
x = \frac{\sigma_3 - \sigma_1}{\sqrt{2}},
\qquad
y = \frac{2\sigma_2 - \sigma_1 - \sigma_3}{\sqrt{6}}
$$

В этих координатах:

$$
\rho = \sqrt{x^2 + y^2} = \sqrt{2 J_2} = \frac{q}{\sqrt{3}}
$$

— расстояние от гидростатической оси до точки;

$$
\theta = \arctan\!\left(\frac{y}{x}\right)
$$

— угол Лоде – Надаи.

### 2. Ориентация осей главных напряжений

Оси $\sigma_1, \sigma_2, \sigma_3$ проецируются на π-плоскость под углами
$+30°, +150°, -90°$ (то есть с шагом $120°$). При этом:

- луч $\theta = +30°$ совпадает с направлением $\sigma_1$ (трёхосное сжатие);
- луч $\theta = -30°$ совпадает с направлением $\sigma_3$ (трёхосное растяжение);
- луч $\theta = 0°$ соответствует чистому сдвигу.

### 3. Сечение Мора – Кулона

При фиксированном $p$ сечение поверхности Мора – Кулона в π-плоскости
представляет собой **неправильный шестиугольник**, описываемый в полярных
координатах:

$$
q(\theta) = \frac{6\sin\varphi \cdot p + 6c\cos\varphi}{3 - \sin\varphi \sin 3\theta},
\qquad
\rho(\theta) = \frac{q(\theta)}{\sqrt{3}}
$$

Вершины шестиугольника чередуются:

- при $\theta = +30°$ (и $-90°$, $+150°$) — «дальние» вершины (сжатие);
- при $\theta = -30°$ (и $+90°$, $-150°$) — «близкие» вершины (растяжение).

### 4. Критерий прочности в π-плоскости

Точка напряжённого состояния находится **внутри** шестиугольника —
прочность обеспечена; **на границе или снаружи** — прочность не обеспечена.

Расстояние от точки до границы можно оценить как:

$$
\Delta \rho = \rho_{env}(\theta) - \rho_{pt}
$$

где $\rho_{env}(\theta)$ — радиус границы MC в направлении текущего угла Лоде,
а $\rho_{pt}$ — радиус точки напряжённого состояния.
"""


# ============================================================
# БЛОК 5. ОСНОВНОЙ РАСЧЁТ
# ============================================================

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

        inv = stress_invariants(sigma)

        sigma_spherical, sigma_deviatoric = split_stress_tensor(sigma)
        epsilon_spherical, epsilon_deviatoric, epsilon = calculate_strain_tensor(sigma, G, K)

        strength_check = check_mohr_coulomb_strength(
            float(eigenvalues[0]), float(eigenvalues[1]), float(eigenvalues[2]), c, phi
        )

        sigma_spherical_principal, sigma_deviatoric_principal = \
            calculate_spherical_and_deviatoric_principal_stresses(eigenvalues)

        # --- 3D-график ---
        show_pyramid = st.session_state.get("show_pyramid", False)
        show_dev_plane = st.session_state.get("show_deviator_plane", False)
        show_mer_plane = st.session_state.get("show_meridian_plane", False)
        update_plot(eigenvalues, c, phi, show_pyramid,
                    show_deviator_plane=show_dev_plane,
                    show_meridian_plane=show_mer_plane)

        # ---------- Гармошка «Результаты расчета» ----------
        with st.expander("Результаты расчета", expanded=False):
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

            st.write("**Инварианты напряжённого состояния:**")
            st.write(f"I₁ = tr σ = {inv['I1']:.2f} кПа")
            st.write(f"J₂ = ½ s:s = {inv['J2']:.2f} кПа²")
            st.write(f"J₃ = ⅓ tr(s³) = {inv['J3']:.2f} кПа³")
            st.write(f"p = I₁/3 = {inv['p']:.2f} кПа")
            st.write(f"q = √(3J₂) = {inv['q']:.2f} кПа")
            st.write(f"Угол Лоде θ = {inv['lode_deg']:.2f}° "f"({inv['lode_rad']:.4f} рад)")

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
        with st.expander("Формулы для расчета (основные)", expanded=False):
            st.markdown(FORMULAS_MAIN)

        # ---------- Гармошка «Девиаторная плоскость» ----------
        with st.expander("Теория: девиаторная (π-) плоскость", expanded=False):
            st.markdown(FORMULAS_DEVIATOR)


        # --- 2D-график в плоскости p–q ---
        update_pq_plot(
            inv, c, phi,
            sigma1=float(eigenvalues[0]),
            sigma3=float(eigenvalues[2]),
        )

        # --- 2D-график в девиаторной плоскости ---
        update_deviator_plot(inv, c, phi, eigenvalues)

    except (ValueError, ZeroDivisionError) as e:
        st.error(f"Проверьте введенные данные! ({e})")


# ============================================================
# БЛОК 6. СИНХРОНИЗАЦИЯ И ИНТЕРФЕЙС
# ============================================================

def sync_symmetric():
    """Синхронизация симметричных компонент тензора."""
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
c1.number_input("σ_xx", value=30.0, key="sigma11", on_change=calculate)
c2.number_input("τ_xy", value=0.0, key="tau12", on_change=sync_symmetric)
c3.number_input("τ_xz", value=0.0, key="tau13", on_change=sync_symmetric)

c1, c2, c3 = st.sidebar.columns(3)
c1.number_input("τ_yx", key="tau21", disabled=True)
c2.number_input("σ_yy", value=30.0, key="sigma22", on_change=calculate)
c3.number_input("τ_yz", value=0.0, key="tau23", on_change=sync_symmetric)

c1, c2, c3 = st.sidebar.columns(3)
c1.number_input("τ_zx", key="tau31", disabled=True)
c2.number_input("τ_zy", key="tau32", disabled=True)
c3.number_input("σ_zz", value=30.0, key="sigma33", on_change=calculate)

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
st.sidebar.checkbox(
    "Показать девиаторную плоскость",
    value=False,
    key="show_deviator_plane",
    on_change=calculate
)
st.sidebar.checkbox(
    "Показать меридианную плоскость",
    value=False,
    key="show_meridian_plane",
    on_change=calculate
)

# ---------- Первичный расчёт ----------
if 'init' not in st.session_state:
    st.session_state.init = True
    calculate()
