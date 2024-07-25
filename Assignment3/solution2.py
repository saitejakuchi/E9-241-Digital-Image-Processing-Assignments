import os

import cv2
from utils import result_folder, image_fft_filter


def solution(path_to_image: str, cut_off_freq: int) -> int:

    img = cv2.imread(path_to_image)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print('Problem-2 Solution :- ')

    image_fft_filter(path_to_image, img, '2', 'Ideal',
                     cut_off_freq=cut_off_freq)
    image_fft_filter(path_to_image, img, '2', 'Gaussian',
                     cut_off_freq=cut_off_freq)


if __name__ == '__main__':
    from utils import problem2_image_path
    os.makedirs(result_folder, exist_ok=True)
    # solution(problem2_image_path, 100)
    for freq in [10, 50, 100, 200]:
        solution(problem2_image_path, freq)
