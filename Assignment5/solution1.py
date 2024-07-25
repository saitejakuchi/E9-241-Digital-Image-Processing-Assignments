import os
from utils import get_pretrained_resnet, knn_classifier, predict_test_data, get_feature_data, results_folder_path


def solution(X_train, y_train, X_test, y_test):
    print(f'Solution-1 Results :- ')
    model = get_pretrained_resnet()
    train_feature_data, test_feature_data = get_feature_data(
        X_train, X_test, model)
    model_knn = knn_classifier(train_feature_data, y_train)
    predict_test_data(test_feature_data, y_test, model_knn, 'Confusion Matrix without fine-tuning', os.path.join(results_folder_path, '1_cm.png'))


if __name__ == '__main__':
    from utils import path_to_images, images_shape, get_train_test_data
    os.makedirs(results_folder_path, exist_ok=True)
    X_train, y_train, X_test, y_test = get_train_test_data(
        path_to_images, images_shape[:2])
    solution(X_train, y_train, X_test, y_test)
