from utils import get_train_test_data, path_to_images, images_shape, results_folder_path
import os

def main():
    solutions_to_run = [1, 2]
    os.makedirs(results_folder_path, exist_ok=True)
    X_train, y_train, X_test, y_test = get_train_test_data(
        path_to_images, images_shape[:2])

    if 1 in solutions_to_run:
        from solution1 import solution
        solution(X_train, y_train, X_test, y_test)

    if 2 in solutions_to_run:
        from solution2 import solution
        solution(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()
