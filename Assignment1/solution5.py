import os

import cv2
from utils import (filter_min_threshold, get_connected_componenets_index,
                   is_within_delta, modified_connected_components,
                   result_folder)


def solution(path_to_image: str) -> None:

    print('Problem-5 Solution :- ')
    # 18-20 for 2 and 4-9 for 1
    delta = 18
    epsilon = 2
    max_percent = 0.04
    min_percent = 0.3

    data = []

    img = cv2.imread(path_to_image)
    if img.shape[2] == 3:
        # Or pick either of the channel data as the pixels are same for every channel.
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # hist_data = get_histogram_data_from_image(gray_img)
    # path_to_image_result = os.path.join('problem5', 'problem5_img_hist_data.png')
    # plot_histogram_data(np.arange(256), hist_data, "Intensity Level", "Frequency",
    #                     f'Histogram data for {path_to_image} image', path_to_image_result)

    number_of_pixels = gray_img.shape[0] * gray_img.shape[1]
    ignore_pixel_threshold = int(max_percent * number_of_pixels)

    for threshold_value in range(255):
        print(f'{threshold_value=}')
        _, bin_img = cv2.threshold(
            gray_img, threshold_value, 255, cv2.THRESH_BINARY)

        result_data = modified_connected_components(
            threshold_value, bin_img, ignore_pixel_threshold)
        result_data = filter_min_threshold(result_data, min_percent)
        data += result_data

    # x_data = [(value[0]) for value in data]
    # y_data = [(value[1]) for value in data]

    # plt.hist(x_data)
    # plt.xlabel('Pixel Index in Image')
    # plt.ylabel('Frequency')
    # plt.title('Pixel index distribution of x-axis for connected components across all threshold values',fontdict={'fontsize': 10})
    # plt.savefig('x_dist.png')
    # plt.close()

    # plt.hist(y_data)
    # plt.xlabel('Pixel Index in Image')
    # plt.ylabel('Frequency')
    # plt.title('Pixel index distribution of y-axis for connected components across all threshold values', fontdict={'fontsize': 10})
    # plt.savefig('y_dist.png')
    # plt.close()

    # filtering for y-cordinate, threshold, size
    filtered_data = [(value[1], value[2], value[3])
                     for value in data if value[0] > 251]

    filtered_data.sort(key=lambda data: data[1])

    threshold_data = {}

    for values in filtered_data:
        threshold_data[values[1]] = threshold_data.get(
            values[1], []) + [(values[0], values[1], values[2])]
    length = len(threshold_data.keys())

    # for finding the proper epsilon and delta values.
    # for delta in range(1, 21):
    #     print(f'For {delta=}')
    #     for epsilon in range(1, 11):
    #         print(f'\tFor {epsilon=}')
    #         stable_threshold_collection = []
    #         for index in range(epsilon, length - epsilon):
    #             # print(index)
    #             data_tuple = threshold_data[index]
    #             prev_data_tuple = threshold_data[index - epsilon]
    #             next_data_tuple = threshold_data[index + epsilon]
    #             stable_thres_values = is_within_delta(
    #                 prev_data_tuple, data_tuple, next_data_tuple, delta)
    #             if stable_thres_values:
    #                 stable_threshold_collection += stable_thres_values
    #         unique_componenet_values = set()
    #         for values in stable_threshold_collection:
    #             unique_componenet_values.add(values[0])
    #         unique_componenet_values = list(unique_componenet_values)
    #         updated_components = get_connected_componenets_index(
    #             unique_componenet_values)
    #         print(f'\t\t{updated_components=}')

    stable_component_collection = []
    for index in range(epsilon, length - epsilon):
        # print(index)
        data_tuple = threshold_data[index]
        prev_data_tuple = threshold_data[index - epsilon]
        next_data_tuple = threshold_data[index + epsilon]
        stable_thres_values = is_within_delta(
            prev_data_tuple, data_tuple, next_data_tuple, delta)
        if stable_thres_values:
            stable_component_collection += stable_thres_values
    unique_componenet_values = set()
    for values in stable_component_collection:
        unique_componenet_values.add(values[0])
    unique_componenet_values = list(unique_componenet_values)
    updated_components = get_connected_componenets_index(
        unique_componenet_values)
    print(f'\t\t{updated_components=}')

    for componenet_data in stable_component_collection:
        for index, values in enumerate(updated_components, 1):
            # import pdb;pdb.set_trace()
            if componenet_data[0] in values:
                print(
                    f'{componenet_data}, For connected component-{index}, Stable threshold = {componenet_data[1]}')
    print(
        f'Number of characters found in the image are {len(updated_components)}')


if __name__ == '__main__':
    from utils import problem5_image_path
    os.makedirs(result_folder, exist_ok=True)
    solution(problem5_image_path)
