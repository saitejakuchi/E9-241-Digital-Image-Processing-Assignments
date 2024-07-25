import os

import matplotlib.pyplot as plt
from skimage.io import imread
from utils import result_folder


def solution(path_to_image: str) -> None:
    gray_img = imread(path_to_image, as_gray=True)
    plt.figure()
    _, axarr = plt.subplots(1, 2)

    axarr[0].imshow(gray_img, cmap='gray')
    axarr[0].title.set_text('Enhanced Image')
    axarr[1].imshow(gray_img, cmap='gray', vmin=0, vmax=255)
    axarr[1].title.set_text('Actual Image')
    image_path = os.path.join(result_folder, '1.png')
    plt.savefig(image_path)
    plt.close()
    print('Problem-1 Solution :- ')
    print(
        f'\tResult image can be found at (wrt current dir) :- {image_path}\n')

if __name__ == '__main__':
    from utils import problem1_image_path
    os.makedirs(result_folder, exist_ok=True)
    solution(problem1_image_path)
