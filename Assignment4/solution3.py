import os

import cv2
from utils import result_folder, hough_transform
import numpy as np


def get_image_data(occlusions=True):
    image_size = (500, 500)
    image = np.zeros(image_size, dtype=np.uint8)

    cv2.line(image, (100, 300), (300, 50), 255, 2)
    cv2.line(image, (300, 50), (300, 400), 255, 5)
    cv2.line(image, (10, 400), (200, 400), 255, 2)
    cv2.line(image, (100, 300), (300, 400), 255, 3)
    if occlusions:
        cv2.circle(image, (300, 100), 50, 255, 2)
    cv2.circle(image, (350, 350), 20, 255, 2)

    return image


def draw_image(data, image, threshold_value, occlusions=True):
    for r, theta in data:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * r
        y0 = b * r
        x1 = int(x0 + 450 * (-b))
        y1 = int(y0 + 450 * (a))
        x2 = int(x0 - 450 * (-b))
        y2 = int(y0 - 450 * (a))
        cv2.line(image, (x1, y1), (x2, y2), 200, 5)
    if occlusions:
        cv2.imwrite(os.path.join(result_folder,
                    f'hough_o_{threshold_value}.png'), image)
    else:
        cv2.imwrite(os.path.join(result_folder,
                    f'hough_{threshold_value}.png'), image)


def solution(path_to_image1) -> None:

    img1 = cv2.imread(path_to_image1)

    if img1.shape[2] == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('gray_img.png', img1)

    image = get_image_data()
    mean = 0
    stddev = 180
    noise = np.zeros(image.shape, np.uint8)
    cv2.randn(noise, mean, stddev)

    noisy_image = cv2.add(image, noise)

    cv2.imwrite(os.path.join(result_folder, '3_w_original_image.png'), image)
    cv2.imwrite(os.path.join(result_folder,
                '3_nw_original_image.png'), noisy_image)

    # point_data_50 = hough_transform(image, 50)
    point_data_100 = hough_transform(image, 100)
    point_data_150 = hough_transform(image, 150)
    # point_data_200 = hough_transform(image, 200)

    # point_ndata_50 = hough_transform(noisy_image, 50)
    # point_ndata_100 = hough_transform(noisy_image, 100)
    # point_ndata_150 = hough_transform(noisy_image, 150)
    # point_ndata_200 = hough_transform(noisy_image, 200)

    # draw_image(point_data_50, image.copy(), 50)
    draw_image(point_data_100, image.copy(), 100)
    draw_image(point_data_150, image.copy(), 150)
    # draw_image(point_data_200, image.copy(), 200)

    # draw_image(point_ndata_50, noisy_image.copy(), 50)
    # draw_image(point_ndata_100, noisy_image.copy(), 100)
    # draw_image(point_ndata_150, noisy_image.copy(), 150)
    # draw_image(point_ndata_200, noisy_image.copy(), 200)

    image = get_image_data(False)
    noisy_image = cv2.add(image, noise)

    cv2.imwrite(os.path.join(result_folder, '3_wo_original_image.png'), image)
    cv2.imwrite(os.path.join(result_folder,
                '3_nwo_original_image.png'), noisy_image)

    # point_data_50 = hough_transform(image, 50)
    point_data_100 = hough_transform(image, 100)
    point_data_150 = hough_transform(image, 150)
    # point_data_200 = hough_transform(image, 200)

    # point_datan_50 = hough_transform(noisy_image, 50)
    # point_datan_100 = hough_transform(noisy_image, 100)
    # point_datan_150 = hough_transform(noisy_image, 150)
    # point_datan_200 = hough_transform(noisy_image, 200)

    # draw_image(point_data_50, image.copy(), 50, False)
    draw_image(point_data_100, image.copy(), 100, False)
    draw_image(point_data_150, image.copy(), 150, False)
    # draw_image(point_data_200, image.copy(), 200, False)

    # draw_image(point_datan_50, noisy_image.copy(), 50, False)
    # draw_image(point_datan_100, noisy_image.copy(), 100, False)
    # draw_image(point_datan_150, noisy_image.copy(), 150, False)
    # draw_image(point_datan_200, noisy_image.copy(), 200, False)


    # point_data_100 = hough_transform(img1, 100)
    # point_data_150 = hough_transform(img1, 150)

    # draw_image(point_data_100, img1.copy(), 100, False)
    # draw_image(point_data_150, img1.copy(), 150, False)

if __name__ == '__main__':
    os.makedirs(result_folder, exist_ok=True)
    from utils import problem3_image1_path
    solution(problem3_image1_path)
