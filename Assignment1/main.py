import os

from utils import result_folder


def main():

    os.makedirs(result_folder, exist_ok=True)
    solutions_to_run = [1, 2, 3, 4, 5]

    if 1 in solutions_to_run:
        from solution1 import solution
        from utils import problem1_image_path
        solution(problem1_image_path)

    if 2 in solutions_to_run:
        from solution2 import solution
        from utils import problem2_image_path
        solution(problem2_image_path)

    if 3 in solutions_to_run:
        from solution3 import solution
        from utils import (problem3_image_background_path,
                           problem3_image_depth_path, problem3_image_text_path)
        solution(problem3_image_text_path, problem3_image_depth_path,
                 problem3_image_background_path)

    if 4 in solutions_to_run:
        from solution4 import solution
        from utils import problem4_image_path
        solution(problem4_image_path)

    if 5 in solutions_to_run:
        from solution5 import solution
        from utils import problem5_image_path
        solution(problem5_image_path)


if __name__ == '__main__':
    main()
