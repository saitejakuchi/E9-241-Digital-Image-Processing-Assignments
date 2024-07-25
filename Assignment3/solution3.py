import os

import cv2
from utils import result_folder, image_fft_filter
from scipy.io import loadmat
import numpy as np


def deblur_image(path_to_image, image_data, image_type, blur_kernel_data):
    problem_number = '3'
    P, Q = image_data.shape
    filter_p, filter_q = blur_kernel_data.shape
    updated_blur_kernel = np.pad(blur_kernel_data, ((
        0, P - filter_p), (0, Q - filter_q)), 'constant', constant_values=0)

    image_fft_filter(path_to_image, image_data, problem_number,
                     'Inverse', image_type, updated_blur_kernel)

    image_fft_filter(path_to_image, image_data, problem_number,
                     'Wiener', image_type, updated_blur_kernel)


def solution(path_to_image1: str, path_to_image2: str, path_to_blur_kernel: str) -> None:

    low_noise_img = cv2.imread(path_to_image1)
    if low_noise_img.shape[2] == 3:
        low_noise_img = cv2.cvtColor(low_noise_img, cv2.COLOR_BGR2GRAY)

    high_noise_img = cv2.imread(path_to_image2)
    if high_noise_img.shape[2] == 3:
        high_noise_img = cv2.cvtColor(high_noise_img, cv2.COLOR_BGR2GRAY)

    blur_kernel = loadmat(path_to_blur_kernel)['h']

    print('Problem-3 Solution :- ')

    deblur_image(path_to_image1, low_noise_img, 'Low', blur_kernel)
    deblur_image(path_to_image2, high_noise_img, 'High', blur_kernel)


if __name__ == '__main__':
    from utils import problem3_image1_path, problem3_image2_path, problem3_blur_file_path
    os.makedirs(result_folder, exist_ok=True)
    solution(problem3_image1_path, problem3_image2_path,
             problem3_blur_file_path)
