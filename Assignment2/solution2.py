import os

import cv2
import numpy as np
from utils import (get_gamma_transform_image, get_histogram_data_from_image,
                   get_normalized_data, histogram_equalization,
                   optimal_gamma_correction, result_folder)


def solution(path_to_image: str) -> int:

    img = cv2.imread(path_to_image)
    if img.shape[2] == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist_data = np.array(get_histogram_data_from_image(gray_img))
    normalized_data = get_normalized_data(
        hist_data, gray_img.shape[0], gray_img.shape[1])
    equalized_histogram_data = histogram_equalization(
        normalized_data)
    # result_file_path = os.path.join(result_folder, '2_gray_hist_cdf.png')
    # plot_cummulative_histogram_wrapper(
    #     gray_img, f'Histogram data for {path_to_image}', result_file_path)

    new_gray_img = equalized_histogram_data[gray_img]
    # result_file1_path = os.path.join(result_folder, '2_hist_eq_cdf.png')
    # plot_cummulative_histogram_wrapper(
    #     new_gray_img, f'Histogram Equalization data for {path_to_image}', result_file1_path)

    hist_path = os.path.join(result_folder,
                             '2_hist_equalized_img.png')
    cv2.imwrite(hist_path, new_gray_img)

    gamma_value = optimal_gamma_correction(gray_img, new_gray_img)
    gamma_transformed_image = get_gamma_transform_image(gray_img, gamma_value)

    # result_file2_path = os.path.join(result_folder, '2_gamma_cdf.png')
    # plot_cummulative_histogram_wrapper(
    #     gamma_transformed_image, f'Gamma Transform data for {path_to_image}', result_file2_path)

    gamma_transform_path = os.path.join(result_folder,
                                        '2_gamma_transform_image.png')

    print('Problem-2 Solution :- ')
    print(
        f'\tHistogram Equalized Image can be found at (wrt current dir) :- {hist_path}')
    print(
        f'\tGamma Transformed Image can be found at (wrt current dir) :- {gamma_transform_path}\n')
    cv2.imwrite(gamma_transform_path, gamma_transformed_image)


if __name__ == '__main__':
    from utils import problem2_image_path
    os.makedirs(result_folder, exist_ok=True)
    solution(problem2_image_path)
