import os
from utils import results_folder_path, get_pretrained_resnet, get_finetuned_model, get_feature_data, knn_classifier, predict_test_data


def solution(X_train, y_train, X_test, y_test):
    print(f'Solution-2 Results (Fine-tuning) :- ')

    base_model = get_pretrained_resnet()
    base_model.trainable = True

    finetuned_model = get_finetuned_model(
        base_model, X_train, y_train, len(set(y_train)))

    train_feature_data, test_feature_data = get_feature_data(
        X_train, X_test, finetuned_model)
    model_knn = knn_classifier(train_feature_data, y_train)
    predict_test_data(test_feature_data, y_test, model_knn, 'Confusion Matrix with fine-tuning', os.path.join(results_folder_path, '2_cm_fine_tune.png'))


if __name__ == '__main__':
    from utils import path_to_images, images_shape, get_train_test_data
    os.makedirs(results_folder_path, exist_ok=True)
    X_train, y_train, X_test, y_test = get_train_test_data(
        path_to_images, images_shape[:2])
    solution(X_train, y_train, X_test, y_test)
