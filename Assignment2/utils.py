import os
import typing

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.ndimage import convolve

result_folder = 'result'
problem1_image_path = 'Images/ECE.png'
problem2_image_path = 'Images/hazy.png'
problem3_image_path = 'Images/box.png'
problem4_image_path = 'Images/study.png'


def get_histogram_data_from_image(img_data: npt.NDArray[typing.Any], as_dict: bool = False) -> typing.List:
    histogram_data = [0] * 256
    for row_index in range(img_data.shape[0]):
        for col_index in range(img_data.shape[1]):
            histogram_data[img_data[row_index][col_index]] += 1
    if as_dict:
        return dict(zip(np.arange(256), histogram_data))
    return histogram_data


def get_normalized_data(histogram_data: npt.NDArray[typing.Any], num_of_rows: int, num_of_cols: int) -> npt.NDArray[typing.Any]:
    cummulative_data = np.cumsum(histogram_data)
    max_value = num_of_rows * num_of_cols
    min_value = np.min(histogram_data[np.nonzero(histogram_data)])
    normalized_data = (cummulative_data - min_value) / (max_value - min_value)
    normalized_data[normalized_data < 0] = 0
    return normalized_data


def plot_cummulative_histogram(label_data: npt.NDArray[typing.Any], freq_data: npt.NDArray[typing.Any], cumm_data: npt.NDArray[typing.Any], xlabel_value: str, ylabel_value: str, title: str, result_img_path: str) -> None:
    plt.bar(label_data, freq_data, label='Frequency Distribution')
    plt.plot(label_data, cumm_data,
             label='Cumulative Distriubtion', color='black')
    plt.xlabel(xlabel_value)
    plt.ylabel(ylabel_value)
    plt.title(title, fontsize=10)
    plt.legend(loc='best')
    plt.savefig(result_img_path)
    plt.close()


def plot_cummulative_histogram_wrapper(image_data: npt.NDArray[typing.Any], title: str, result_path: str) -> None:
    hist_data = np.array(get_histogram_data_from_image(image_data))
    normalized_data = get_normalized_data(
        hist_data, image_data.shape[0], image_data.shape[1])
    normalized_data = np.round(normalized_data * max(hist_data))
    plot_cummulative_histogram(np.arange(256), hist_data, normalized_data, 'Intensity Level',
                               'Frequency', title, result_path)


def plot_error_distriubtion(label_data: npt.NDArray[typing.Any], freq_data: npt.NDArray[typing.Any], xlabel_value: str, ylabel_value: str, title: str, result_img_path: str) -> None:
    plt.plot(label_data, freq_data)
    plt.xlabel(xlabel_value)
    plt.ylabel(ylabel_value)
    plt.title(title)
    plt.yscale('log')
    plt.savefig(result_img_path)
    plt.close()


def histogram_equalization(normalized_data: npt.NDArray[typing.Any]) -> npt.NDArray[typing.Any]:
    cummulative_data = np.round(normalized_data * 255)
    cummulative_data = cummulative_data.astype(np.uint8)
    return cummulative_data


def mean_square_error_value(actual_data: npt.NDArray[typing.Any], predicted_data: npt.NDArray[typing.Any]) -> float:
    return np.mean(np.power(actual_data - predicted_data, 2))


def get_gamma_transform_image(image_data: npt.NDArray[typing.Any], gamma_value: float) -> npt.NDArray[typing.Any]:
    return np.array(255 * (image_data / 255) ** gamma_value, dtype=np.int64)


def optimal_gamma_correction(img_data: npt.NDArray[typing.Any], hist_eq_img_data: npt.NDArray[typing.Any]) -> float:
    error_values = []
    for gamma_value in np.arange(0.1, 5.1, 0.1):
        gamma_transformed_image = get_gamma_transform_image(
            img_data, gamma_value)
        mse_error = mean_square_error_value(
            hist_eq_img_data, gamma_transformed_image)
        error_values.append(mse_error)
    # result_img_path = os.path.join(result_folder, '2_gamma_mse_dist.png')
    # plot_error_distriubtion(np.arange(0.1, 5.1, 0.1), error_values,
    #                         'Gamma Value', 'MSE Value', 'MSE Error Distriubtion', result_img_path)
    return (np.argmin(error_values) + 1) / 10


def get_filter(size: int) -> npt.NDArray[typing.Any]:
    return np.ones((size, size)) / (size ** 2)


def get_blur_image(image_data: npt.NDArray[typing.Any], filter_size: int) -> npt.NDArray[typing.Any]:
    filter_data = get_filter(filter_size)
    return convolve(image_data, filter_data, mode='reflect', cval=0.0)


def generate_plot_data(image_data: npt.NDArray[typing.Any], filter_size: int, title: str, keyword: str, save_image: bool = True, problem_number: int = 4) -> None:
    # hist_file_path = os.path.join(
    #     result_folder, f'{problem_number}_{keyword}_{filter_size}_hist_cdf.png')
    # plot_cummulative_histogram_wrapper(
    #     image_data, title, hist_file_path)
    if save_image:
        result_path = os.path.join(
            result_folder, f'{problem_number}_{keyword}_{filter_size}_img.png')
        print(
            f'\tHigh Boost Filtered Image with filter size {filter_size} can be found at (wrt current dir) :- {result_path}')
        cv2.imwrite(result_path, image_data)


def generate_blur_image_data(path_to_image: str, gray_image: npt.NDArray[typing.Any], filter_size: int, problem_number: int = 4) -> npt.NDArray[typing.Any]:
    blur_img = get_blur_image(gray_image, filter_size)
    generate_plot_data(blur_img, filter_size,
                       f'Histogram data for Blur Image with filter_size = {filter_size} of {path_to_image}', 'blur', False, problem_number)
    return blur_img


def generate_high_boost_image(path_to_image: str, gray_image: npt.NDArray[typing.Any], blur_img: npt.NDArray[typing.Any], k: float, filter_size: int, problem_number: int = 4) -> None:
    mask = k * cv2.subtract(gray_image, blur_img)
    mask = mask.astype(np.uint8)
    high_boost_img = cv2.add(gray_image, mask).astype(np.uint8)
    # high_boost_img = cv2.addWeighted(gray_image, k + 1, blur_img, -1 * k, 0)
    if k == 1:
        keyword = 'unsharp'
        title = f'Histogram data for Unsharp Image with filter_size = {filter_size} of {path_to_image}'
    else:
        keyword = 'high_boost'
        title = f'Histogram data for High Boost Image with filter_size = {filter_size} of {path_to_image}'
    generate_plot_data(high_boost_img, filter_size,
                       title, keyword, problem_number)


def rotate_image(image_data: npt.NDArray[typing.Any], degree: int, interpolation_type: str) -> npt.NDArray[typing.Any]:
    height, width = image_data.shape[:2]

    radians = np.radians(degree)
    cos_value, sin_value = np.cos(radians), np.sin(radians)

    rotated_height = round(height * np.abs(cos_value) +
                           width * np.abs(sin_value))
    rotated_width = round(width * np.abs(cos_value) +
                          height * np.abs(sin_value))
    rotated_image = np.zeros(
        (rotated_height, rotated_width, image_data.shape[2])).astype(np.uint8)

    original_center_x, original_center_y = width // 2, height // 2
    center_x, center_y = rotated_width // 2, rotated_height // 2

    for x_rotated in range(rotated_height):
        for y_rotated in range(rotated_width):
            if degree < 0:
                x_original = (x_rotated - center_x) * cos_value - \
                    (y_rotated - center_y) * sin_value + original_center_y
                y_original = (x_rotated - center_x) * sin_value + \
                    (y_rotated - center_y) * cos_value + original_center_x
            else:
                x_original = (x_rotated - center_x) * cos_value - \
                    (y_rotated - center_y) * sin_value + original_center_x
                y_original = (x_rotated - center_x) * sin_value + \
                    (y_rotated - center_y) * cos_value + original_center_y

            if interpolation_type == 'nn':
                x_original = int(x_original + 0.5)
                y_original = int(y_original + 0.5)
                if (0 <= x_original < height) and (0 <= y_original < width):
                    rotated_image[x_rotated, y_rotated,
                                  :] = image_data[x_original, y_original, :]

            if interpolation_type == 'bilin':
                if (0 <= x_original < height - 1) and (0 <= y_original < width - 1):
                    x1_original = int(x_original)
                    y1_original = int(y_original)
                    x2_original = x1_original + 1
                    y2_original = y1_original + 1
                    a, b = (x_original - x1_original), (y_original - y1_original)
                    data1 = image_data[x1_original, y1_original, :]
                    data2 = image_data[x1_original, y2_original, :]
                    data3 = image_data[x2_original, y1_original, :]
                    data4 = image_data[x2_original, y2_original, :]
                    data = (1 - a) * (1 - b) * data1 + (1 - a) * b * \
                        data2 + (1 - b) * a * data3 + \
                        a * b * data4
                    rotated_image[x_rotated, y_rotated, :] = data
    return rotated_image


# def convolution(image_data: npt.NDArray[typing.Any], filter_data: npt.NDArray[typing.Any], filter_size: int) -> npt.NDArray[typing.Any]:
#     padding_value = filter_size // 2
#     result_img_data = image_data.copy()
#     new_image_data = np.pad(
#         image_data, (padding_value, padding_value), mode='edge')
#     for row_index in range(padding_value, padding_value + image_data.shape[0]):
#         for col_index in range(padding_value, padding_value + image_data.shape[1]):
#             sub_image = new_image_data[row_index - padding_value:row_index +
#                                        padding_value + 1, col_index - padding_value:col_index + padding_value + 1]
#             conv_result = np.sum(np.multiply(
#                 sub_image, filter_data))
#             result_img_data[row_index - padding_value,
#                             col_index - padding_value] = conv_result

#     result_img_data = np.round(result_img_data).astype(np.uint8)
#     return result_img_data


# def get_blur_image_old(image_data: npt.NDArray[typing.Any], filter_size: int) -> npt.NDArray[typing.Any]:
#     filter_data = get_filter(filter_size)
#     return convolution(image_data, filter_data, filter_size)


# def generate_masked_image_old(path_to_image, gray_image, blur_image, filter_size):
#     masked_img = cv2.subtract(gray_image, blur_image)
#     generate_plot_data(masked_img, filter_size,
#                        f'Histogram data for Masked Image with filter_size = {filter_size} of {path_to_image}', 'masked', False)
#     return masked_img


# def generate_high_boost_image_old(path_to_image, gray_image, masked_image, k, filter_size, problem_number=4):
#     high_pass_component = np.round(k * masked_image).astype('uint8')
#     high_boost_img = cv2.add(gray_image, high_pass_component)
#     if k == 1:
#         keyword = 'unsharp'
#         title = f'Histogram data for Unsharp Image with filter_size = {filter_size} of {path_to_image}'
#     else:
#         keyword = 'high_boost'
#         title = f'Histogram data for High Boost Image with filter_size = {filter_size} of {path_to_image}'
#     generate_plot_data(high_boost_img, filter_size,
#                        title, keyword, problem_number)

# def rotate_image_matrix(image_data, degree, interpolation_type):

#     radians = np.radians(degree)
#     cos_value, sin_value = np.cos(radians), np.sin(radians)

#     height, width = image_data.shape[:2]
#     rotated_height = round(height * np.abs(cos_value) +
#                            width * np.abs(sin_value))
#     rotated_width = round(width * np.abs(cos_value) +
#                           height * np.abs(sin_value))
#     rotated_image = np.zeros(
#         (rotated_height, rotated_width, image_data.shape[2])).astype(np.uint8)
#     original_center_x, original_center_y = width // 2, height // 2
#     center_x, center_y = rotated_width // 2, rotated_height // 2

#     rotation_matrix = np.array(
#         [[cos_value, -sin_value], [sin_value, cos_value]])

#     indices_data = np.indices((rotated_height, rotated_width)).reshape(
#         2, -1).astype(np.int64)
#     indices_data_transform = indices_data.copy()
#     indices_data_transform[0] = indices_data_transform[0] - center_x
#     indices_data_transform[1] = indices_data_transform[1] - center_y

#     transformed_indices = np.dot(rotation_matrix, indices_data_transform)
#     transformed_indices[0] = transformed_indices[0] + original_center_x
#     transformed_indices[1] = transformed_indices[1] + original_center_y

#     if interpolation_type == 'nn':
#         transformed_indices = np.ceil(transformed_indices).astype(np.int64)
#         valid_indices = np.all(
#             (transformed_indices[1] >= 0, transformed_indices[1] < width, transformed_indices[0] >= 0, transformed_indices[0] < height), axis=0)
#         rotated_image[indices_data[0][valid_indices], indices_data[1][valid_indices], :
#                       ] = image_data[transformed_indices[0][valid_indices], transformed_indices[1][valid_indices], :]

#     if interpolation_type == 'bilin':
#         int_transformed_indices = transformed_indices.astype(np.int64)
#         next_transformed_indices = int_transformed_indices.copy()
#         next_transformed_indices[0] = next_transformed_indices[0] + 1
#         next_transformed_indices[1] = next_transformed_indices[1] + 1
#         valid_indices = np.all(
#             (transformed_indices[1] >= 0, transformed_indices[1] < width - 1, transformed_indices[0] >= 0, transformed_indices[0] < height - 1), axis=0)
#         a, b = transformed_indices[0] - \
#             int_transformed_indices[0], transformed_indices[1] - \
#             int_transformed_indices[1]
#         a, b = a[valid_indices], b[valid_indices]
#         a = np.stack([a, a, a], axis=-1)
#         b = np.stack([b, b, b], axis=-1)
#         data1 = image_data[int_transformed_indices[0][valid_indices],
#                            int_transformed_indices[1][valid_indices], :]
#         data2 = image_data[int_transformed_indices[0][valid_indices],
#                            next_transformed_indices[1][valid_indices], :]
#         data3 = image_data[next_transformed_indices[0][valid_indices],
#                            int_transformed_indices[1][valid_indices], :]
#         data4 = image_data[next_transformed_indices[0][valid_indices],
#                            next_transformed_indices[1][valid_indices], :]
#         data = (1 - a) * (1 - b) * data1 + (1 - a) * b * \
#             data2 + a * (1 - b) * data3 + a * b * data4
#         rotated_image[indices_data[0][valid_indices],
#                       indices_data[1][valid_indices], :] = data

#     return rotated_image
