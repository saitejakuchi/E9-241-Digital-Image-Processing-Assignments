import numpy as np
from scipy.ndimage import convolve
import cv2
import os
import math

result_folder = 'Results'

problem1_image_path = 'Images/building_noisy.png'

problem2_image1_path = 'Images/book_noisy1.png'
problem2_image2_path = 'Images/book_noisy2.png'
problem2_image3_path = 'Images/architecture_noisy1.png'
problem2_image4_path = 'Images/architecture_noisy2.png'

problem3_image1_path = 'Images/test.png'


# reference :- https://jblindsay.github.io/ghrg/Whitebox/Help/FilterLaplacian.html


def get_laplcian_filter(size):
    if size == 3:
        return np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0],
        ])
    return np.array([
        [0, 0, -1, 0, 0],
        [0, -1, -2, -1, 0],
        [-1, -2, 17, -2, -1],
        [0, -1, -2, -1, 0],
        [0, 0, -1, 0, 0],
    ])


def get_gaussian_blur_kernel(size, sigma):

    kernel_data = np.zeros((size, size))

    for i in range(0, size):
        for j in range(0, size):
            kernel_data[i, j] = np.sqrt(
                abs(i - size // 2) ** 2 + abs(j - size // 2) ** 2)

    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((kernel_data / sigma) ** 2) * 0.5)


def get_blur_image(image_data, size, sigma=10):
    kernel_data = get_gaussian_blur_kernel(size, sigma)
    return convolve(image_data, kernel_data, mode='reflect', cval=0.0)


def get_patch(img, x, y, size):
    half = size // 2
    return img[x - half: x + half + 1, y - half: y + half + 1]


def get_bilateral_image(image_data, size, spatial_sigma, intensity_sigma):

    result = np.zeros(image_data.shape)
    height, width = image_data.shape

    spatial_filter = get_gaussian_blur_kernel(size, spatial_sigma)

    for index in range(size // 2, height - size // 2):
        for index1 in range(size // 2, width - size // 2):

            patch_data = get_patch(image_data, index, index1, size)
            intensity_diff = patch_data - image_data[index, index1]
            intensity_gauss = 1 / (intensity_sigma * np.sqrt(2 * np.pi)) * \
                np.exp(-((intensity_diff / intensity_sigma) ** 2) * 0.5)

            filtered_data = intensity_gauss * spatial_filter
            result[index, index1] = np.sum(
                patch_data * filtered_data) / np.sum(filtered_data)
    return result


def laplacian_filter(image_data, filter_applied):
    filter_3 = get_laplcian_filter(3)

    result1 = cv2.filter2D(image_data, -1, filter_3)
    return np.abs(result1).astype(np.uint8)


def get_gradient_kernel(name):
    if name == 'robert':
        return np.array([[1, 0], [0, -1]]), np.array([[0, 1], [-1, 0]])

    constant_value = 1
    if name == 'sobel':
        constant_value = 2
    X = np.array([
        [-1, 0, 1],
        [-constant_value, 0, constant_value],
        [-1, 0, 1]]
    )
    Y = np.array([
        [-1, -constant_value, -1],
        [0, 0, 0],
        [1, constant_value, 1]
    ])
    return X, Y


def detect_edges(image_data, gradient_name, number, size, threshold_value=20):
    gradientX, gradientY = get_gradient_kernel(gradient_name)
    x1_gradient = cv2.filter2D(image_data, -1, gradientX)
    y1_gradient = cv2.filter2D(image_data, -1, gradientY)
    edge1_data = np.abs(x1_gradient) + np.abs(y1_gradient)
    _, thresholded_image1 = cv2.threshold(
        edge1_data, threshold_value, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(result_folder,
                f'2_img{number}_ker_{gradient_name}_smt_{size}_thr_{threshold_value}.png'), thresholded_image1)

def hough_transform(image_data, threshold_value):
    height, width = image_data.shape
    distance = int(np.round(np.sqrt(height**2 + width**2)))
    
    thetas = np.deg2rad(np.arange(-90, 90))
    rs = np.arange(-distance, distance)

    accumulator = np.zeros((len(rs), len(thetas)))

    y, x = np.nonzero(image_data)

    for index in range(len(x)):
        for index1 in range(len(thetas)):
            r = int(x[index] * np.cos(thetas[index1]) + y[index] * np.sin(thetas[index1]))
            r_index = np.argmin(np.abs(rs - r))
            accumulator[r_index, index1] += 1

    result_data = []

    for i in range(accumulator.shape[0]):
        for j in range(accumulator.shape[1]):
            if accumulator[i, j] > threshold_value:
                r = rs[i]
                theta = thetas[j]
                result_data.append((r, theta))

    return result_data