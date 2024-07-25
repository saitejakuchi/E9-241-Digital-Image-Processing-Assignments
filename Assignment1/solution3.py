import os

import cv2
import numpy as np
from utils import otsu_binarization, result_folder


def solution(path_to_text: str, path_to_depth: str, path_to_background: str) -> None:
    # threshold choosen from manual analysis of depth image.
    depth_img = cv2.imread(path_to_depth)
    text_img = cv2.imread(path_to_text)
    background_img = cv2.imread(path_to_background)

    if depth_img.shape[2] == 3:
        # Or pick either of the channel data as the pixels are same for every channel.
        gray_depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
    result_img_path = os.path.join(result_folder, 'dept_image_hist.png')
    threshold_value = otsu_binarization(
        path_to_depth, result_img_path, gray_depth_img, False, False)

    _, bin_img = cv2.threshold(
        gray_depth_img, threshold_value, 255, cv2.THRESH_BINARY)
    # cv2.imwrite('problem3_binarized_image.png', bin_img)
    inv_bin_img = cv2.bitwise_not(bin_img)

    # stacking so that it can be applied as filter to colored images.
    filter_img = np.stack([bin_img, bin_img, bin_img], axis=-1)
    inv_filter_img = np.stack([inv_bin_img, inv_bin_img, inv_bin_img], axis=-1)

    text_extracted_image = cv2.bitwise_and(filter_img, text_img)
    background_removed_image = cv2.bitwise_and(background_img, inv_filter_img)

    # cv2.imwrite('problem3_text_extract.png', text_extracted_image)
    # cv2.imwrite('problem3_background_removed.png', background_removed_image)

    final_img = text_extracted_image + background_removed_image

    path_to_image_result = os.path.join(result_folder, 'problem3.png')
    print(
        f'Problem-3 Solution :- \n\tResult Image can be found at (wrt current dir) :- {path_to_image_result}\n')

    cv2.imwrite(path_to_image_result, final_img)


if __name__ == '__main__':
    from utils import (problem3_image_background_path,
                       problem3_image_depth_path, problem3_image_text_path)
    os.makedirs(result_folder, exist_ok=True)
    solution(problem3_image_text_path, problem3_image_depth_path,
             problem3_image_background_path)
