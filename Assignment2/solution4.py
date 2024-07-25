import os

import cv2
from utils import result_folder, generate_blur_image_data, generate_high_boost_image


def solution(path_to_image: str) -> None:

    print('Problem-4 Solution :- ')

    img = cv2.imread(path_to_image)
    if img.shape[2] == 3:
        # Or pick either of the channel data as the pixels are same for every channel.
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # result_file_path = os.path.join(result_folder, '4_gray_hist_cdf.png')

    # plot_cummulative_histogram_wrapper(
    #     gray_img, f'Histogram data for {path_to_image}', result_file_path)

    filter_size = 5
    blur_img_5 = generate_blur_image_data(path_to_image, gray_img, filter_size)

    generate_high_boost_image(path_to_image, gray_img,
                              blur_img_5, 2.5, filter_size)

    filter_size = 3

    blur_img_3 = generate_blur_image_data(path_to_image, gray_img, filter_size)

    generate_high_boost_image(path_to_image, gray_img,
                              blur_img_3, 2.5, filter_size)
    print()


if __name__ == '__main__':
    from utils import problem4_image_path
    os.makedirs(result_folder, exist_ok=True)
    solution(problem4_image_path)
