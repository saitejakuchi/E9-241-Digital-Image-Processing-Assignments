import os

import cv2
from utils import (find_connected_components,
                   get_character_count_without_punctuations, otsu_binarization,
                   result_folder)


def solution(path_to_image: str) -> None:

    img = cv2.imread(path_to_image)
    if img.shape[2] == 3:
        # Or pick either of the channel data as the pixels are same for every channel.
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result_img_path = os.path.join(result_folder, 'problem4_hist.png')
    threshold_value = otsu_binarization(
        path_to_image, result_img_path, gray_img, False, False)

    _, bin_img = cv2.threshold(
        gray_img, threshold_value, 255, cv2.THRESH_BINARY_INV)
    # cv2.imwrite('problem4_binarized_img.png', bin_img)

    region_based_img, total_characters_found = find_connected_components(
        bin_img)

    max_pixel_threshold = 0.01*(gray_img.shape[0]*gray_img.shape[1])
    actual_character_count = get_character_count_without_punctuations(
        region_based_img, total_characters_found, max_pixel_threshold)
    print(
        f'Problem-4 Solution :- \n\tCharacter count excluding punctuations is {actual_character_count}\n')


if __name__ == '__main__':
    from utils import problem4_image_path
    os.makedirs(result_folder, exist_ok=True)
    solution(problem4_image_path)
