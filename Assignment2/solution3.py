import os

import cv2
from utils import result_folder, rotate_image


def solution(path_to_image: str) -> None:

    img = cv2.imread(path_to_image)

    rotated_image_5_nn = rotate_image(img, 5, 'nn')
    rotated_image_30_nn = rotate_image(img, -30, 'nn')
    rotated_image_5_bi = rotate_image(img, 5, 'bilin')
    rotated_image_30_bi = rotate_image(img, -30, 'bilin')

    nn_c5_image_path = os.path.join(result_folder,
                                    '3_c5_nn_rotated_img.png')
    bi_c5_image_path = os.path.join(result_folder,
                                    '3_c5_bi_rotated_img.png')
    nn_cc30_image_path = os.path.join(result_folder,
                                      '3_cc30_nn_rotated_img.png')
    bi_cc30_image_path = os.path.join(result_folder,
                                      '3_cc30_bi_rotated_img.png')

    cv2.imwrite(nn_c5_image_path, rotated_image_5_nn)
    cv2.imwrite(nn_cc30_image_path, rotated_image_30_nn)
    cv2.imwrite(bi_c5_image_path, rotated_image_5_bi)
    cv2.imwrite(bi_cc30_image_path, rotated_image_30_bi)

    print('Problem-3 Solution :- ')
    print(
        f'\tClockwise 5 degree NN Interpolation Image can be found at (wrt current dir) :- {nn_c5_image_path}')
    print(
        f'\tClockwise 5 degree Bi-linear Interpolation Image can be found at (wrt current dir) :- {bi_c5_image_path}')
    print(
        f'\tCounter-Clockwise 30 degree NN Interpolation Image can be found at (wrt current dir) :- {nn_cc30_image_path}')
    print(
        f'\tCounter-Clockwise 30 degree Bi-linear Interpolation Image can be found at (wrt current dir) :- {bi_cc30_image_path}\n')


if __name__ == '__main__':
    from utils import problem3_image_path
    os.makedirs(result_folder, exist_ok=True)
    solution(problem3_image_path)
