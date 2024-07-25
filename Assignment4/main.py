import os

from utils import result_folder


def main():

    os.makedirs(result_folder, exist_ok=True)
    solutions_to_run = [1, 2, 3]

    if 1 in solutions_to_run:
        from solution1 import solution
        from utils import problem1_image_path
        solution(problem1_image_path)

    if 2 in solutions_to_run:
        from solution2 import solution
        from utils import problem2_image1_path, problem2_image2_path, problem2_image3_path, problem2_image4_path
        solution(problem2_image1_path, problem2_image2_path,
                 problem2_image3_path, problem2_image4_path)

    if 3 in solutions_to_run:
        from solution3 import solution
        from utils import problem3_image1_path, problem3_image2_path
        solution(problem3_image1_path, problem3_image2_path)


if __name__ == '__main__':
    main()
