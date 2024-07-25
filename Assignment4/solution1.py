import os
from utils import result_folder, get_blur_image, get_bilateral_image, laplacian_filter
import cv2
import matplotlib.pyplot as plt


def solution(path_to_image):
    img_data = cv2.imread(path_to_image)

    if img_data.shape[2] == 3:
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)

    img_smooth = get_blur_image(img_data, 7, 21)
    cv2.imwrite(os.path.join(result_folder, f'1_gaussian_19.png'), img_smooth)

    # for sigma in range(60, 80):
    #     img_bi_smooth = get_bilateral_image(img_data, 7, sigma, sigma)
    #     cv2.imwrite(f'1_{sigma}.png', img_bi_smooth)

    spatial_std, intensity_std = 9, 9
    bilater_img = get_bilateral_image(img_data, 7, spatial_std, intensity_std)
    cv2.imwrite(os.path.join(result_folder,
                f'1_bilateral_{spatial_std}_{intensity_std}.png'), bilater_img)

    edge_data = laplacian_filter(img_data, 'normal')
    edge_data1 = laplacian_filter(bilater_img, 'bi-lateral')
    edge_data2 = laplacian_filter(img_smooth, 'gauss-7')


    result_data = cv2.threshold(
        edge_data, 140, 255, cv2.THRESH_BINARY)[1]
    result_data1 = cv2.threshold(
        edge_data1, 140, 255, cv2.THRESH_BINARY)[1]

    cv2.imwrite(os.path.join(result_folder,
                f'normal_image_edge_140.png'), result_data)
    cv2.imwrite(os.path.join(result_folder,
                f'bi_image_edge_140.png'), result_data1)

    result_data = cv2.threshold(
        edge_data, 159, 255, cv2.THRESH_BINARY)[1]
    result_data1 = cv2.threshold(
        edge_data1, 159, 255, cv2.THRESH_BINARY)[1]

    cv2.imwrite(os.path.join(result_folder,
                f'normal_image_edge_159.png'), result_data)
    cv2.imwrite(os.path.join(result_folder,
                f'bi_image_edge_159.png'), result_data1)


    result_data2 = cv2.threshold(
        edge_data2, 4, 255, cv2.THRESH_BINARY)[1]

    cv2.imwrite(os.path.join(result_folder,
                f'gauss_image_edge_4.png'), result_data2)


if __name__ == '__main__':
    from utils import problem1_image_path
    os.makedirs(result_folder, exist_ok=True)
    solution(problem1_image_path)
