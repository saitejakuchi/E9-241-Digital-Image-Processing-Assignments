import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from keras.applications.resnet50 import ResNet50
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

path_to_images = 'Images'
results_folder_path = 'Results/'

images_shape = (224, 224, 3)
base_learning_rate = 0.00001
epochs = 10
validation_split = 0.1
random_num = 41
batch_size = 32


def get_image_data(path_to_image, img_shape):
    img_data = cv2.imread(path_to_image)
    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    img_data = cv2.resize(img_data, img_shape)
    return img_data


def get_train_test_data(path_to_data, image_shape):
    X_train_data, X_test_data, y_train_data, y_test_data = [], [], [], []
    for folder_name in os.listdir(path_to_data):
        classtype, datatype = folder_name.split('_')
        folder_path = os.path.join(path_to_data, folder_name)
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image_data = get_image_data(image_path, image_shape)
            if datatype == 'test':
                X_test_data += [image_data]
                y_test_data += [classtype]
            if datatype == 'train':
                X_train_data += [image_data]
                y_train_data += [classtype]
    return np.array(X_train_data), np.array(y_train_data), np.array(X_test_data), np.array(y_test_data)


def get_pretrained_resnet():
    return ResNet50(include_top=False, weights='imagenet',
                    input_shape=images_shape)


def get_feature_data(X_train, X_test, model):
    train_feature_data, test_feature_data = model.predict(
        X_train), model.predict(X_test)
    train_shape, test_shape = train_feature_data.shape, test_feature_data.shape
    train_feature_data, test_feature_data = np.reshape(
        train_feature_data, (train_shape[0], np.prod(train_shape[1:]))), np.reshape(test_feature_data, (test_shape[0], np.prod(test_shape[1:])))
    return train_feature_data, test_feature_data


def get_finetuned_model(base_model, X_train, y_train, unique_classes):
    new_model = base_model.output
    new_model = tf.keras.layers.GlobalAveragePooling2D()(new_model)
    new_model = tf.keras.layers.Dense(
        unique_classes, activation='softmax')(new_model)

    finetune_model = tf.keras.Model(inputs=base_model.input, outputs=new_model)

    finetune_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                           loss=tf.keras.losses.CategoricalCrossentropy(),
                           metrics=['accuracy'])

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=validation_split, random_state=random_num)

    finetune_model.fit(x=X_train, y=pd.get_dummies(y_train), batch_size=batch_size,
                       epochs=epochs, validation_data=(X_valid, pd.get_dummies(y_valid)))

    return tf.keras.Model(
        finetune_model.input, finetune_model.get_layer('conv5_block3_out').output)


def knn_classifier(X_train, y_train, k=3):
    model = KNeighborsClassifier(k)
    model.fit(X_train, y_train)
    return model


def predict_test_data(X_test, y_test, model, title, result_image_path):
    label_data = list(set(y_test))
    y_predict = model.predict(X_test)
    accuracy_value = accuracy_score(y_test, y_predict)
    confusion_matrix_data = confusion_matrix(
        y_pred=y_predict, y_true=y_test, labels=label_data)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix_data, display_labels=label_data)
    disp.plot()
    plt.title(title)
    print(
        f'Accuracy Value is {accuracy_value*100}%\nConfusion Matrix is :- \n{confusion_matrix_data}')
    plt.savefig(result_image_path, bbox_inches='tight')
    plt.close()
