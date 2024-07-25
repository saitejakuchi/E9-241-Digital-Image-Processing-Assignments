import os
import cv2
from utils import result_folder, get_blur_image, detect_edges


def solution(path_to_image1, path_to_image2, path_to_image3, path_to_image4):
    img1 = cv2.imread(path_to_image1)
    img2 = cv2.imread(path_to_image2)
    img3 = cv2.imread(path_to_image3)
    img4 = cv2.imread(path_to_image4)

    if img1.shape[2] == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    if img2.shape[2] == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if img3.shape[2] == 3:
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

    if img4.shape[2] == 3:
        img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)

    img1_3_smooth = get_blur_image(img1, 3)
    img1_5_smooth = get_blur_image(img1, 5)
    # img1_7_smooth = get_blur_image(img1, 7)

    img2_3_smooth = get_blur_image(img2, 3)
    img2_5_smooth = get_blur_image(img2, 5)
    # img2_7_smooth = get_blur_image(img2, 7)

    img3_3_smooth = get_blur_image(img3, 3)
    img3_5_smooth = get_blur_image(img3, 5)
    # img3_7_smooth = get_blur_image(img3, 7)

    img4_3_smooth = get_blur_image(img4, 3)
    img4_5_smooth = get_blur_image(img4, 5)
    # img4_7_smooth = get_blur_image(img4, 7)

    # detect_edges(img1_3_smooth, 'robert', 1, 3, 20)

    detect_edges(img1_3_smooth, 'sobel', 1, 3, 5)
    detect_edges(img1_3_smooth, 'prewit', 1, 3, 5)

    detect_edges(img1_3_smooth, 'sobel', 1, 3, 10)
    detect_edges(img1_3_smooth, 'prewit', 1, 3, 10)

    detect_edges(img1_3_smooth, 'sobel', 1, 3, 20)
    detect_edges(img1_3_smooth, 'prewit', 1, 3, 20)


    # detect_edges(img1_5_smooth, 'robert', 1, 5, 20)
    detect_edges(img1_5_smooth, 'sobel', 1, 5, 5)
    detect_edges(img1_5_smooth, 'prewit', 1, 5, 5)

    detect_edges(img1_5_smooth, 'sobel', 1, 5, 10)
    detect_edges(img1_5_smooth, 'prewit', 1, 5, 10)

    detect_edges(img1_5_smooth, 'sobel', 1, 5, 20)
    detect_edges(img1_5_smooth, 'prewit', 1, 5, 20)

    # detect_edges(img1_7_smooth, 'robert', 1, 7, 20)
    # detect_edges(img1_7_smooth, 'sobel', 1, 7, 20)
    # detect_edges(img1_7_smooth, 'prewit', 1, 7, 20)

    # detect_edges(img2_3_smooth, 'robert', 2, 3, 20)
    detect_edges(img2_3_smooth, 'sobel', 2, 3, 5)
    detect_edges(img2_3_smooth, 'prewit', 2, 3, 5)

    detect_edges(img2_3_smooth, 'sobel', 2, 3, 10)
    detect_edges(img2_3_smooth, 'prewit', 2, 3, 10)

    detect_edges(img2_3_smooth, 'sobel', 2, 3, 20)
    detect_edges(img2_3_smooth, 'prewit', 2, 3, 20)

    # detect_edges(img2_5_smooth, 'robert', 2, 5, 20)
    detect_edges(img2_5_smooth, 'sobel', 2, 5, 5)
    detect_edges(img2_5_smooth, 'prewit', 2, 5, 5)

    detect_edges(img2_5_smooth, 'sobel', 2, 5, 10)
    detect_edges(img2_5_smooth, 'prewit', 2, 5, 10)

    detect_edges(img2_5_smooth, 'sobel', 2, 5, 20)
    detect_edges(img2_5_smooth, 'prewit', 2, 5, 20)

    # detect_edges(img2_7_smooth, 'robert', 2, 7, 20)
    # detect_edges(img2_7_smooth, 'sobel', 2, 7, 20)
    # detect_edges(img2_7_smooth, 'prewit', 2, 7, 20)

    # detect_edges(img3_3_smooth, 'robert', 3, 3, 20)
    detect_edges(img3_3_smooth, 'sobel', 3, 3, 5)
    detect_edges(img3_3_smooth, 'prewit', 3, 3, 5)

    detect_edges(img3_3_smooth, 'sobel', 3, 3, 10)
    detect_edges(img3_3_smooth, 'prewit', 3, 3, 10)

    detect_edges(img3_3_smooth, 'sobel', 3, 3, 20)
    detect_edges(img3_3_smooth, 'prewit', 3, 3, 20)

    # detect_edges(img3_5_smooth, 'robert', 3, 5, 20)
    detect_edges(img3_5_smooth, 'sobel', 3, 5, 5)
    detect_edges(img3_5_smooth, 'prewit', 3, 5, 5)

    detect_edges(img3_5_smooth, 'sobel', 3, 5, 10)
    detect_edges(img3_5_smooth, 'prewit', 3, 5, 10)

    detect_edges(img3_5_smooth, 'sobel', 3, 5, 20)
    detect_edges(img3_5_smooth, 'prewit', 3, 5, 20)

    # detect_edges(img3_7_smooth, 'robert', 3, 7, 20)
    # detect_edges(img3_7_smooth, 'sobel', 3, 7, 20)
    # detect_edges(img3_7_smooth, 'prewit', 3, 7, 20)

    # detect_edges(img4_3_smooth, 'robert', 4, 3, 20)
    detect_edges(img4_3_smooth, 'sobel', 4, 3, 5)
    detect_edges(img4_3_smooth, 'prewit', 4, 3, 5)

    detect_edges(img4_3_smooth, 'sobel', 4, 3, 10)
    detect_edges(img4_3_smooth, 'prewit', 4, 3, 10)

    detect_edges(img4_3_smooth, 'sobel', 4, 3, 20)
    detect_edges(img4_3_smooth, 'prewit', 4, 3, 20)


    # detect_edges(img4_5_smooth, 'robert', 4, 5, 20)
    detect_edges(img4_5_smooth, 'sobel', 4, 5, 5)
    detect_edges(img4_5_smooth, 'prewit', 4, 5, 5)

    detect_edges(img4_5_smooth, 'sobel', 4, 5, 10)
    detect_edges(img4_5_smooth, 'prewit', 4, 5, 10)

    detect_edges(img4_5_smooth, 'sobel', 4, 5, 20)
    detect_edges(img4_5_smooth, 'prewit', 4, 5, 20)

    # detect_edges(img4_7_smooth, 'robert', 4, 7, 20)
    # detect_edges(img4_7_smooth, 'sobel', 4, 7, 20)
    # detect_edges(img4_7_smooth, 'prewit', 4, 7, 20)


if __name__ == '__main__':
    from utils import problem2_image1_path, problem2_image2_path, problem2_image3_path, problem2_image4_path
    os.makedirs(result_folder, exist_ok=True)
    solution(problem2_image1_path, problem2_image2_path,
             problem2_image3_path, problem2_image4_path)
