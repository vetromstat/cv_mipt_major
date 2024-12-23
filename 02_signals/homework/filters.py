import numpy as np


def conv_nested(image, kernel):
    """Наивная реализация свертки."""
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # Дополняем изображение нулями
    pad_height = Hk // 2
    pad_width = Wk // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Проходим по каждому пикселю изображения
    for i in range(Hi):
        for j in range(Wi):
            # Применяем ядро к текущему пикселю
            for m in range(Hk):
                for n in range(Wk):
                    out[i, j] += padded_image[i + m, j + n] * kernel[m, n]

    return out

def zero_pad(image, pad_height, pad_width):
    """Дополнение изображения нулями."""
    H, W = image.shape
    out = np.zeros((H + 2 * pad_height, W + 2 * pad_width))
    out[pad_height:pad_height + H, pad_width:pad_width + W] = image
    return out

def conv_fast(image, kernel):
    """Эффективная реализация свертки."""
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # Дополняем изображение
    pad_height = Hk // 2
    pad_width = Wk // 2
    padded_image = zero_pad(image, pad_height, pad_width)

    # Переворачиваем ядро
    kernel_flipped = np.flip(kernel)

    # Применяем свертку
    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = np.sum(padded_image[i:i + Hk, j:j + Wk] * kernel_flipped)

    return out

def conv_faster(image, kernel):
    """Более быстрая реализация свертки."""
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # Дополняем изображение
    pad_height = Hk // 2
    pad_width = Wk // 2
    padded_image = zero_pad(image, pad_height, pad_width)

    # Переворачиваем ядро
    kernel_flipped = np.flip(kernel)

    # Используем векторизацию для ускорения свертки
    for i in range(Hi):
        for j in range(Wi):
            # Извлекаем область изображения, соответствующую текущей позиции
            region = padded_image[i:i + Hk, j:j + Wk]
            # Выполняем свертку с использованием векторизации
            out[i, j] = np.sum(region * kernel_flipped)

    return out

def cross_correlation(f, g):
    """Кросс-корреляция f и g."""
    return conv_fast(f, np.flip(g))

def zero_mean_cross_correlation(f, g):
    """Кросс-корреляция с нулевым средним значением."""
    g_mean = np.mean(g)
    g_zero_mean = g - g_mean
    return cross_correlation(f, g_zero_mean)

def normalized_cross_correlation(f, g):
    """Нормализованная кросс-корреляция f и g."""
    g_mean = np.mean(g)
    g_std = np.std(g)
    g_normalized = (g - g_mean) / g_std

    out = np.zeros_like(f)
    Hf, Wf = f.shape
    Hg, Wg = g.shape

    pad_height = Hg // 2
    pad_width = Wg // 2
    padded_f = zero_pad(f, pad_height, pad_width)

    for i in range(Hf):
        for j in range(Wf):
            patch = padded_f[i:i + Hg, j:j + Wg]
            patch_mean = np.mean(patch)
            patch_std = np.std(patch)

            if patch_std > 0:  # Избегаем деления на ноль
                patch_normalized = (patch - patch_mean) / patch_std
                out[i, j] = np.sum(patch_normalized * g_normalized)

    return out