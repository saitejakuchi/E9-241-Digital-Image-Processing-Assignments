import typing
from collections import deque
from time import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

result_folder = 'result'
problem1_image_path = 'Images/coins.png'
problem2_image_path = 'Images/coins.png'
problem3_image_text_path = 'Images/IIScText.png'
problem3_image_background_path = 'Images/IIScMainBuilding.png'
problem3_image_depth_path = 'Images/IIScTextDepth.png'
problem4_image_path = 'Images/quote.png'
problem5_image_path = 'Images/Characters.png'


def get_histogram_data_from_image(img_data: npt.NDArray[typing.Any], as_dict: bool = False) -> typing.List:
    histogram_data = [0] * 256
    for row_index in range(img_data.shape[0]):
        for col_index in range(img_data.shape[1]):
            histogram_data[img_data[row_index][col_index]] += 1
    if as_dict:
        return dict(zip(np.arange(256), histogram_data))
    return histogram_data


def plot_histogram_data(label_data: npt.NDArray[typing.Any], freq_data: npt.NDArray[typing.Any], x_label: str, y_label: str, title: str, result_file_path: str) -> None:
    plt.bar(label_data, freq_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(result_file_path)


def plot_characters_size_distribution(data: npt.NDArray[typing.Any]):
    plt.hist(data)
    plt.title('Character Size Distribution')
    plt.xlabel('Character Size')
    plt.ylabel('Number of Characters')
    plt.savefig('character_size_distriubtion.png')


def compute_intra_class_variance(intensty_data: npt.NDArray[typing.Any], pixel_count: int, threshold: int, max_intensity_value: int = 256) -> float:

    background_pixel_data = intensty_data[:threshold]
    if background_pixel_data.size:
        background_intensity_count = np.sum(background_pixel_data)
        background_intensity_mask = np.arange(threshold)
        background_intensity_weights = background_intensity_count/pixel_count
        if background_intensity_count:
            background_intensity_mean = np.sum(
                background_intensity_mask * background_pixel_data)/background_intensity_count
            background_intensity_variance = np.sum(
                (background_intensity_mask - background_intensity_mean) ** 2 * background_pixel_data)/background_intensity_count
        else:
            # Handling edge case where intensity count is zero.
            background_intensity_mean = 0
            background_intensity_variance = 0
    else:
        # Handling edge case where intensity count is zero.
        background_intensity_weights = 0
        background_intensity_mean = 0
        background_intensity_variance = 0

    foreground_pixel_data = intensty_data[threshold:]
    if foreground_pixel_data.size:
        foreground_intensity_count = np.sum(foreground_pixel_data)
        foreground_intensity_mask = np.arange(threshold, max_intensity_value)
        foreground_intensity_weights = foreground_intensity_count/pixel_count
        if foreground_intensity_count:
            foreground_intensity_mean = np.sum(
                foreground_intensity_mask * foreground_pixel_data)/foreground_intensity_count
            foreground_intensity_variance = np.sum(
                (foreground_intensity_mask - foreground_intensity_mean) ** 2 * foreground_pixel_data)/foreground_intensity_count
        else:
            # Handling edge case where intensity count is zero.
            foreground_intensity_mean = 0
            foreground_intensity_variance = 0
    else:
        # Handling edge case where intensity count is zero.
        foreground_intensity_weights = 0
        foreground_intensity_mean = 0
        foreground_intensity_variance = 0

    assert (foreground_intensity_weights + background_intensity_weights == 1)
    intra_class_variance = background_intensity_weights * background_intensity_variance + \
        foreground_intensity_weights * foreground_intensity_variance
    return intra_class_variance


def compute_inter_class_variance(intensty_data: npt.NDArray[typing.Any], pixel_count: int, threshold: int, max_intensity_value: int = 256) -> float:

    background_pixel_data = intensty_data[:threshold]
    if background_pixel_data.size:
        background_intensity_count = np.sum(background_pixel_data)
        background_intensity_mask = np.arange(threshold)
        background_intensity_weights = background_intensity_count/pixel_count
        if background_intensity_count:
            background_intensity_mean = np.sum(
                background_intensity_mask * background_pixel_data)/background_intensity_count
        else:
            # Handling edge case where intensity count is zero.
            background_intensity_mean = 0
    else:
        # Handling edge case where intensity count is zero.
        background_intensity_weights = 0
        background_intensity_mean = 0

    foreground_pixel_data = intensty_data[threshold:]
    if foreground_pixel_data.size:
        foreground_intensity_count = np.sum(foreground_pixel_data)
        foreground_intensity_mask = np.arange(threshold, max_intensity_value)
        foreground_intensity_weights = foreground_intensity_count/pixel_count
        if foreground_intensity_count:
            foreground_intensity_mean = np.sum(
                foreground_intensity_mask * foreground_pixel_data)/foreground_intensity_count
        else:
            # Handling edge case where intensity count is zero.
            foreground_intensity_mean = 0
    else:
        # Handling edge case where intensity count is zero.
        foreground_intensity_weights = 0
        foreground_intensity_mean = 0

    assert (foreground_intensity_weights + background_intensity_weights == 1)
    inter_class_variance = background_intensity_weights * foreground_intensity_weights * \
        (background_intensity_mean - foreground_intensity_mean) ** 2
    return inter_class_variance


def otsu_binarization(image_file_path: str, result_img_path: str, image_data: npt.NDArray[typing.Any], print_stats_data: bool = False, save_hist: bool = False, intensity_level: int = 257) -> int:
    intra_variance_results = []
    inter_variance_results = []

    intensity_data = np.array(get_histogram_data_from_image(image_data))
    if save_hist:
        plot_histogram_data(np.arange(intensity_level - 1), intensity_data, "Intensity Level",
                            "Frequency", f'Histogram data for {image_file_path} image', result_img_path)
    total_pixels = image_data.shape[0] * image_data.shape[1]

    start = time()
    for threshold in range(intensity_level):
        inter_variance = compute_inter_class_variance(
            intensity_data, total_pixels, threshold, intensity_level - 1)
        inter_variance_results.append(inter_variance)
    intra_time = time() - start

    start = time()
    for threshold in range(intensity_level):
        intra_variance = compute_intra_class_variance(
            intensity_data, total_pixels, threshold, intensity_level - 1)
        intra_variance_results.append(intra_variance)
    inter_time = time() - start

    intra_variance_results = np.round(np.array(intra_variance_results), 6)
    inter_variance_results = np.round(np.array(inter_variance_results), 6)

    intra_variance_threshold = np.argmin(intra_variance_results)
    intra_variance_value = np.min(intra_variance_results)

    inter_variance_threshold = np.argmax(inter_variance_results)
    inter_variance_value = np.max(inter_variance_results)

    threshold_sum = np.array(intra_variance_results) + \
        np.array(inter_variance_results)
    threshold_sum = np.round(threshold_sum, 5)

    global_threshold_mask = np.arange(256)
    global_threshold_intensity_count = np.sum(intensity_data)
    global_threshold_mean = np.sum(
        global_threshold_mask * intensity_data)/global_threshold_intensity_count
    global_threshold_variance = np.round(np.sum(
        (global_threshold_mask - global_threshold_mean)**2 * intensity_data)/global_threshold_intensity_count, 5)

    global_threshold_array = np.array(
        [global_threshold_variance] * intensity_level)

    # plots for the distriubtion of variance calculated across various threshold values

    # plt.plot(np.arange(257), inter_variance_results)
    # plt.xlabel('Intensity Level')
    # plt.ylabel('Between-class variance')
    # plt.title('Distribution of between-class variance for various threshold values.')
    # plt.savefig(os.path.join(result_folder, 'inter_class_dist.png'))
    # plt.close()

    # plt.plot(np.arange(257), intra_variance_results)
    # plt.xlabel('Intensity Level')
    # plt.ylabel('Within-class variance')
    # plt.title('Distribution of within-class variance for various threshold values.')
    # plt.savefig(os.path.join(result_folder, 'intra_class_dist.png'))
    # plt.close()

    # plt.plot(np.arange(257), global_threshold_array)
    # plt.xlabel('Intensity Level')
    # plt.ylabel('Global variance')
    # plt.title('Distribution of Global variance for various threshold values.')
    # plt.savefig(os.path.join(result_folder, 'global_var_dist.png'))

    np.testing.assert_array_equal(global_threshold_array, threshold_sum)
    if print_stats_data:
        print(f'\tGlobal Threshold Variance = {global_threshold_variance}')
        print('\tAsserted inter-class variance + intra-class variance = global thresholding value, for every threshold value t = 0 to 256')
    assert (intra_variance_threshold == inter_variance_threshold)
    if print_stats_data:
        print('\tAsserted inter-class threshold == intra-class threshold')
    optimal_threshold = inter_variance_threshold
    if print_stats_data:
        print(
            f'\tOptimal threshold value = {optimal_threshold}')
        print(
            f'\tIntra class variance (at t={optimal_threshold}) = {intra_variance_value}')
        print(
            f'\tInter class variance  (at t={optimal_threshold}) = {inter_variance_value}')
        print(f'\tInter-variance calculation took around {intra_time} secs.')
        print(f'\tIntra-variance calculation took around {inter_time} secs.')
        print(f'\tSpeedup = {inter_time/intra_time}\n')

    return optimal_threshold


def BFS(image_data: npt.NDArray[typing.Any], row_index: int, col_index: int, rows: int, columns: int, region_num: int) -> None:

    queue_data = deque()
    image_data[row_index][col_index] = region_num
    queue_data.append((row_index, col_index))

    while queue_data:

        curr_row_index, curr_col_index = queue_data.popleft()

        if (curr_row_index > 0) and (image_data[curr_row_index - 1][curr_col_index] == -1):
            image_data[curr_row_index - 1][curr_col_index] = region_num
            queue_data.append((curr_row_index - 1, curr_col_index))  # top

        if (curr_col_index > 0) and (image_data[curr_row_index][curr_col_index - 1] == -1):
            image_data[curr_row_index][curr_col_index - 1] = region_num
            queue_data.append((curr_row_index, curr_col_index - 1))  # left

        if (curr_row_index < rows - 1) and (image_data[curr_row_index + 1][curr_col_index] == -1):
            image_data[curr_row_index + 1][curr_col_index] = region_num
            queue_data.append((curr_row_index + 1, curr_col_index))  # bottom

        if (curr_col_index < columns - 1) and (image_data[curr_row_index][curr_col_index + 1] == -1):
            image_data[curr_row_index][curr_col_index + 1] = region_num
            queue_data.append((curr_row_index, curr_col_index + 1))  # right


def modified_BFS(image_data: npt.NDArray[typing.Any], row_index: int, col_index: int, rows: int, columns: int, region_num: int) -> int:

    queue_data = deque()
    image_index_data = image_data[row_index][col_index]
    image_data[row_index][col_index] = region_num
    queue_data.append((row_index, col_index))
    pixel_count = 0

    while queue_data:

        pixel_count += 1
        curr_row_index, curr_col_index = queue_data.popleft()

        if (curr_row_index > 0) and (image_data[curr_row_index - 1][curr_col_index] < 1) and (image_index_data == image_data[curr_row_index - 1][curr_col_index]):
            image_data[curr_row_index - 1][curr_col_index] = region_num
            queue_data.append((curr_row_index - 1, curr_col_index))  # top

        if (curr_col_index > 0) and (image_data[curr_row_index][curr_col_index - 1] < 1) and (image_index_data == image_data[curr_row_index][curr_col_index - 1]):
            image_data[curr_row_index][curr_col_index - 1] = region_num
            queue_data.append((curr_row_index, curr_col_index - 1))  # left

        if (curr_row_index < rows - 1) and (image_data[curr_row_index + 1][curr_col_index] < 1) and (image_index_data == image_data[curr_row_index + 1][curr_col_index]):
            image_data[curr_row_index + 1][curr_col_index] = region_num
            queue_data.append((curr_row_index + 1, curr_col_index))  # bottom

        if (curr_col_index < columns - 1) and (image_data[curr_row_index][curr_col_index + 1] < 1) and (image_index_data == image_data[curr_row_index][curr_col_index + 1]):
            image_data[curr_row_index][curr_col_index + 1] = region_num
            queue_data.append((curr_row_index, curr_col_index + 1))  # right
    return pixel_count


def get_character_count_without_punctuations(image_data: npt.NDArray[typing.Any], current_character_count: int, ignore_pixel_threshold: int) -> int:

    count_data = get_histogram_data_from_image(image_data, True)

    # filtering for non-zero intensity pixels.
    count_data = dict(filter(lambda item: item[1] != 0, count_data.items()))
    # print(f'{count_data=}')

    filtered_character_data = dict(
        filter(lambda item: item[1] < ignore_pixel_threshold, count_data.items()))

    # if assertion error then need to fix the dfs algorithm.
    assert (len(filtered_character_data.keys()) == current_character_count)
    minimum_threshold_value = max(filtered_character_data.values())*0.2

    # plot_characters_size_distribution(filtered_character_data.values())

    # filtering values which are only above and below certain threshold
    filtered_character_data = dict(
        filter(lambda item: item[1] > minimum_threshold_value, filtered_character_data.items()))

    # print(f'{filtered_character_data=}')
    # print(set(count_data.keys()) - set(filtered_character_data.keys()))

    return len(filtered_character_data.keys())


def find_connected_components(image_data: npt.NDArray[typing.Any]) -> (npt.NDArray[typing.Any], int):

    # copying image data into new variable and binarizing it.
    result_img = image_data.copy()
    result_img = result_img//-255
    rows, columns = result_img.shape

    characters_count = 0
    for row_index in range(rows):
        for col_index in range(columns):
            if result_img[row_index][col_index] == -1:
                characters_count += 1
                BFS(result_img, row_index, col_index,
                    rows, columns, characters_count)

    # DEBUG PURPOSE
    # data = Counter(result_img.flatten())
    # for key, _ in data.items():
    #     copy_result_img = result_img.copy()
    #     copy_result_img[(copy_result_img != 0) & (copy_result_img == key)] = 255
    #     copy_result_img[copy_result_img != 255] = 0
    #     cv2.imwrite(f'temp_{key}.png', copy_result_img)

    return result_img, characters_count


def modified_connected_components(threshold_val: int, image_data: npt.NDArray[typing.Any], max_thres: int) -> (npt.NDArray[typing.Any], int):

    result_img = image_data.copy()
    result_img = result_img//-255
    rows, columns = result_img.shape

    cluster_input_data = []

    characters_count = 0
    for row_index in range(rows):
        for col_index in range(columns):
            if result_img[row_index][col_index] < 1:
                characters_count += 1
                character_size = modified_BFS(result_img, row_index, col_index,
                                              rows, columns, characters_count)
                if character_size < max_thres:
                    cluster_input_data.append(
                        (row_index, col_index, threshold_val, character_size))

    return cluster_input_data


def filter_min_threshold(data, percent):
    sizes = []
    for values in data:
        sizes.append(values[3])
    min_threshold = int(max(sizes)*percent)
    new_result_data = []
    for values in data:
        if values[3] > min_threshold:
            new_result_data.append(values)
    return new_result_data


def get_connected_componenets_index(threshold_data):
    final_result = []
    for value in threshold_data:
        similar_points = set()
        similar_points.add(value)
        for value1 in threshold_data:
            diff = abs(value1 - value)
            if diff > 0 and diff < 11:
                similar_points.add(value1)
        if similar_points not in final_result:
            final_result.append(similar_points)
    return final_result


def is_within_delta(prev_thres_data, curr_thres_data, next_thres_data, delta_value):
    stable_threshold_data = []
    for curr_thres_value in curr_thres_data:
        prev_within_delta = False
        next_within_delta = False
        for prev_thres_value in prev_thres_data:
            prev_within_delta = prev_within_delta or (
                abs(prev_thres_value[2] - curr_thres_value[2]) <= delta_value)
        for next_thres_value in next_thres_data:
            next_within_delta = next_within_delta or (
                abs(next_thres_value[2] - curr_thres_value[2]) <= delta_value)
        if prev_within_delta and next_within_delta:
            stable_threshold_data.append(curr_thres_value)
    return stable_threshold_data
