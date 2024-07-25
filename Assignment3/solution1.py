import os
import numpy as np
from utils import result_folder, plot_freq_data, plot_filter_data


def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def solution(freq, M=500):
    temp_freq = (2 * np.pi * freq)/M

    print('Problem-1 Solution :- ')

    img = np.fromfunction(lambda i, j: np.cos(
        (temp_freq * distance(i, j, M/2, M/2))), (M, M))
    result_img_path = os.path.join(
        result_folder, f'1_{freq}_sinusoidal_image.png')
    plot_filter_data(img, result_img_path,
                     f'Sinusoidal Image with Frequency f0 = {freq}')
    print(
        f'\tResult image can be found at (wrt current dir) :- {result_img_path}\n')

    img_dft = np.fft.fft2(img)
    result_img_path = os.path.join(
        result_folder, f'1_{freq}_freq_domain.png')
    plot_freq_data(img_dft, result_img_path,
                   f'DFT of Sinusoidal Image with Frequency f0 = {freq}')
    print(
        f'\tResult image can be found at (wrt current dir) :- {result_img_path}\n')

    idft_img = np.fft.ifft2(img_dft).real
    result_img_path = os.path.join(
        result_folder, f'1_{freq}_recovered_spatial_domain.png')
    plot_filter_data(idft_img, result_img_path,
                     f'Recovered Sinusoidal Image with Frequency f0 = {freq}')
    print(
        f'\tResult image can be found at (wrt current dir) :- {result_img_path}\n')


if __name__ == '__main__':
    os.makedirs(result_folder, exist_ok=True)
    for freq_value in [10, 50, 100]:
        solution(freq_value)
