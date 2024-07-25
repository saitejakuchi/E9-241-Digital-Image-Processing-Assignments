import os

import cv2
import numpy as np
from utils import (get_histogram_data_from_image, plot_histogram_data,
                   result_folder)


def solution(path_to_image: str) -> None:
    img = cv2.imread(path_to_image)
    if img.shape[2] == 3:
        # Or pick either of the channel data as the pixels are same for every channel.
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    number_of_pixels = gray_img.shape[0] * gray_img.shape[1]

    hist_data = get_histogram_data_from_image(gray_img)
    intensity_levels = np.arange(256)
    avg_intensity = np.sum(gray_img) / number_of_pixels
    avg_intensity_histogram = sum(
        hist_data * intensity_levels) / number_of_pixels
    print('Problem-1 Solution :- ')
    print(f'\t{avg_intensity=}, {avg_intensity_histogram=}')
    path_to_image_result = os.path.join(result_folder, 'problem1.png')
    plot_histogram_data(intensity_levels, hist_data, "Intensity Level", "Frequency",
                        f'Histogram data for {path_to_image} image', path_to_image_result)
    print(
        f'\tResult image can be found at (wrt current dir) :- {path_to_image_result}\n')


if __name__ == '__main__':
    from utils import problem1_image_path
    os.makedirs(result_folder, exist_ok=True)
    solution(problem1_image_path)
