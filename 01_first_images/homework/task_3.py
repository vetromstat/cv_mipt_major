import cv2
import numpy as np


def rotate(image: np.ndarray, center: tuple, angle: float) -> np.ndarray:
    """
    Повернуть изображение вокруг заданной точки на заданный угол.
    
    :param image: изображение для поворота
    :param center: точка поворота
    :param angle : угол поворота в градусах
    :return: повернутое изображение
    """
    # Получаем размеры изображения
    height, width = image.shape[:2]
    
    # Получаем матрицу поворота
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Вычисляем новые размеры изображения
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    
    # Корректируем матрицу поворота
    rotation_matrix[0, 2] += new_width / 2 - width / 2
    rotation_matrix[1, 2] += new_height / 2 - height / 2
    
    # Поворачиваем изображение
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
    
    return rotated_image

def apply_warpAffine(image: np.ndarray, points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    Применить аффинное преобразование к изображению.
    
    :param image: исходное изображение
    :param points1: исходные точки
    :param points2: целевые точки
    :return: преобразованное изображение
    """
    # Получаем матрицу аффинного преобразования
    matrix = cv2.getAffineTransform(points1, points2)
    
    # Получаем размеры изображения
    height, width = image.shape[:2]
    
    # Применяем аффинное преобразование
    transformed_image = cv2.warpAffine(image, matrix, (width, height))
    
    return transformed_image
