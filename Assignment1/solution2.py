import os

import cv2
from utils import otsu_binarization, result_folder


def solution(path_to_image: str) -> int:
    print(f'Problem-2 Solution :- ')
    img = cv2.imread(path_to_image)
    if img.shape[2] == 3:
        # Or pick either of the channel data as the pixels are same for every channel.
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result_img_path = os.path.join(result_folder, 'problem2_hist.png')
    otsu_binarization(
        path_to_image, result_img_path, gray_img, True, False)

    # image output for result file.
    # _, binarized_image = cv2.threshold(
    #     gray_img, threshold_value, 255, cv2.THRESH_BINARY)
    # cv2.imwrite('problem2_binarized_image.png', binarized_image)


if __name__ == '__main__':
    from utils import problem2_image_path
    os.makedirs(result_folder, exist_ok=True)
    solution(problem2_image_path)
