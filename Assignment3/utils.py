import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

result_folder = 'result'

problem2_image_path = 'Images/characters.tif'
problem3_image1_path = 'Images/Blurred_LowNoise.png'
problem3_image2_path = 'Images/Blurred_HighNoise.png'
problem3_blur_file_path = 'Images/BlurKernel.mat'


def plot_freq_data(freq_data, result_image_path, title):
    plt.imshow(np.log(1 + np.abs(np.fft.fftshift(freq_data))), cmap='gray')
    plt.title(title)
    plt.savefig(result_image_path, bbox_inches='tight')
    plt.close()


def plot_filter_data(data, result_image_path, title):
    plt.imshow(data, cmap='gray')
    plt.title(title)
    plt.savefig(result_image_path, bbox_inches='tight')
    plt.close()


def get_filter(problem_no, row_size, col_size, filter_name, image_type, blur_filter_kernel, cut_off_freq, threshold_value, signal_k):
    if problem_no == '2':
        if filter_name == 'Ideal':
            filter_data = np.fromfunction(lambda i, j: ((
                (i - row_size/2) ** 2)+((j - col_size/2) ** 2)) <= (cut_off_freq ** 2), (row_size, col_size)) * 1
        else:
            filter_data = np.fromfunction(
                lambda i, j: np.exp(-((i - row_size/2) ** 2+(j - col_size/2) ** 2)/(2 * cut_off_freq ** 2)), (row_size, col_size))
        # result_filter_freq = os.path.join(
        #     result_folder, f'{problem_no}_{cut_off_freq}_{filter_name}_freq_domain.png')
        # plot_filter_data(filter_data, result_filter_freq,
        #                  f'{filter_name} low-pass filter with D0={cut_off_freq}')
    else:

        sigma_value = 10
        if image_type == 'low':
            sigma_value = 1

        if filter_name == 'Inverse':
            filter_dft = np.fft.fft2(blur_filter_kernel)
            shifted_filter_dft = np.fft.fftshift(filter_dft)
            inverse_filter_dft = 1/shifted_filter_dft
            inverse_filter_dft[np.abs(shifted_filter_dft)
                               < threshold_value] = 0
            filter_data = inverse_filter_dft.copy()
        else:
            filter_dft = np.fft.fftshift(np.fft.fft2(blur_filter_kernel))

            u, v = np.meshgrid(range(-row_size // 2, row_size // 2),
                               range(-col_size // 2, col_size // 2))
            inv_snr = sigma_value * (np.sqrt(u ** 2 + v ** 2)) / signal_k
            conj_filter_dft = np.conjugate(filter_dft)
            magnitude_dft = np.abs(filter_dft) ** 2

            filter_data = conj_filter_dft / (magnitude_dft + inv_snr)

    return filter_data


def image_fft_filter(path_to_image, image_data, problem_number, filter_name, image_type=None, blur_kernel=None, cut_off_freq=100, threshold=0.1, k=10 ** 5):

    P, Q = image_data.shape
    image_dft = np.fft.fft2(image_data)
    shifted_image_dft = np.fft.fftshift(image_dft)

    # if image_type:
    #     result_img_freq = os.path.join(
    #         result_folder, f'{problem_number}_{image_type}_image_freq_domain.png')
    # else:
    #     result_img_freq = os.path.join(
    #         result_folder, f'{problem_number}_image_freq_domain.png')

    # plot_freq_data(image_dft, result_img_freq,
    #                f'DFT of Image :- {path_to_image}')

    filter_dft = get_filter(problem_number, P, Q,
                            filter_name, image_type, blur_kernel, cut_off_freq, threshold, k)

    filter_image = shifted_image_dft * filter_dft

    filtered_result_image = np.fft.ifft2(
        np.fft.ifftshift(filter_image)).real
    if image_type is None:
        result_image_path = os.path.join(
            result_folder, f'{problem_number}_{cut_off_freq}_{filter_name}_result_img.png')

        recovered_filter = np.fft.ifftshift(np.fft.ifft2(filter_dft))
        recovered_result_image_path = os.path.join(
            result_folder, f'{problem_number}_{cut_off_freq}_{filter_name}_recovered_img.png')
        plot_filter_data(np.abs(recovered_filter), recovered_result_image_path,
                         f'Filter Data in Spatial Domain')

    else:
        result_image_path = os.path.join(
            result_folder, f'{problem_number}_{threshold}_{filter_name}_{image_type}_result_img.png')

    cv2.imwrite(result_image_path, filtered_result_image)
    print(
        f'\tResult image can be found at (wrt current dir) :- {result_image_path}\n')
