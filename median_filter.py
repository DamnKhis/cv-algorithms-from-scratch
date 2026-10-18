import os

import cv2
import numpy as np


def median_filter(image, kernel_size):
    shape = image.shape
    height, width = shape[:2]
    is_color = len(shape) == 3

    pad = kernel_size // 2

    padded_img = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    output = np.zeros_like(image)

    for i in range(height):
        for j in range(width):

            window = padded_img[i: i + kernel_size, j: j + kernel_size]

            if is_color:

                median_value = np.median(window, axis=(0, 1))
            else:

                median_value = np.median(window)

            output[i, j] = median_value

    return output


def main():
    file_path = input("Введите имя файла изображения (например, image.jpg): ")

    if not os.path.exists(file_path):
        print(f"Ошибка: Файл '{file_path}' не найден.")
        return

    img = cv2.imread(file_path)

    if img is None:
        print("Ошибка: Не удалось загрузить изображение.")
        return

    try:
        k_size = int(input("Введите размер ядра (нечетнре число): "))
        if k_size % 2 == 0 or k_size < 1:
            print("Ошибка: Размер ядра должен быть положительным НЕЧЕТНЫМ числом.")
            return
    except ValueError:
        print("Ошибка: Введите целое число.")
        return

    print(f"Обработка {file_path}...")

    denoised_img = median_filter(img, k_size)

    cv2.imshow('Original', img)
    cv2.imshow(f'Result (Kernel {k_size})', denoised_img)

    print("Готово! Нажмите любую клавишу для выхода.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
