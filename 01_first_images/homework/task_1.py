import cv2
import numpy as np
from collections import deque

def find_way_from_maze(image: np.ndarray) -> tuple:
    """
    Найти путь через лабиринт от верхней части до нижней.
    
    :param image: изображение лабиринта
    :return: координаты пути (x, y)
    """
    # Преобразуем изображение в черно-белое
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Определяем размеры изображения
    height, width = binary.shape

    # Находим все возможные входы и выходы
    starts = [(0, x) for x in range(width) if binary[0, x] == 255]
    ends = [(height - 1, x) for x in range(width) if binary[height - 1, x] == 255]

    if not starts or not ends:
        return (np.array([]), np.array([]))  # Если нет входов или выходов

    # BFS для поиска пути
    queue = deque(starts)
    visited = np.zeros((height, width), dtype=bool)
    parent = np.full((height, width), None)  # Массив для хранения родительских узлов

    for start in starts:
        visited[start] = True

    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # вниз, вправо, вверх, влево

    found_exit = False
    exit_point = None

    while queue:
        current = queue.popleft()
        if current in ends:
            found_exit = True
            exit_point = current
            break

        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            if (0 <= neighbor[0] < height and
                0 <= neighbor[1] < width and
                binary[neighbor] == 255 and
                not visited[neighbor]):
                visited[neighbor] = True
                parent[neighbor] = current
                queue.append(neighbor)

    # Восстанавливаем путь
    path = []
    if found_exit:
        step = exit_point
        while step is not None:
            path.append(step)
            step = parent[step[0], step[1]]  # Используем индексы для доступа к родителю
        path.reverse()

    # Разделяем координаты
    if path:
        x_coords, y_coords = zip(*path)
        return (np.array(x_coords), np.array(y_coords))
    else:
        return (np.array([]), np.array([]))  # Если путь не найден