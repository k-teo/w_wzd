import cv2
from keras import Model
from tensorflow import keras
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from os import listdir
from os.path import isfile, join

image_size = (224, 224)
# images_folder = r'../tsne/images/music_instruments_images'
images_folder = r'../tsne/images/valid_dest_50'
# images_folder = r'../tsne/images/new_images'


def get_features(model, image_path, img_size, scalar, offset):
    img = plt.imread(image_path)
    display_img = img / 255.0
    img = cv2.resize(img, image_size)
    img = img / scalar + offset
    img = np.expand_dims(img, axis=0)
    extract = Model(model.inputs, model.layers[-2].output)
    prediction = extract.predict(img)
    index = np.argmax(prediction)
    prob = prediction[0][index] * 100
    return prob, display_img, prediction


def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range


def load_model():
    model_path = '../tsne/model/model.h5'
    return keras.models.load_model(model_path)


def get_model_summary():
    model.summary()


def get_random_colors(length):
    colors = []
    for j in range(length):
        colors.append("#" + ''.join([random.choice('ABCDEF0123456789') for _ in range(6)]))
    return colors


def get_all_photos_paths_from_file(path):
    return [f for f in listdir(path) if isfile(join(path, f))]


def transform_to_tsne(predictions):
    return TSNE(n_components=2, perplexity=5).fit_transform(predictions)


def predict_with_features(model, photo_paths):
    predictions = []
    for path in photo_paths:
        full_photo_path = images_folder + '/' + path
        try:
            image = cv2.imread(full_photo_path)
            image_height, image_width, _ = image.shape
            prob, display_img, prediction = get_features(model, full_photo_path, (image_height, image_width), 1, 0)
            predictions.append(prediction[0])
        except:
            print(full_photo_path)
    return predictions


def scale_image(image, max_image_size):
    image_height, image_width, _ = image.shape

    scale = max(1, image_width / max_image_size, image_height / max_image_size)
    image_width = int(image_width / scale)
    image_height = int(image_height / scale)

    image = cv2.resize(image, (image_width, image_height))
    return image


def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_height, image_width, _ = image.shape

    center_x = int(image_centers_area_size * x) + offset
    center_y = int(image_centers_area_size * (1 - y)) + offset

    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)

    br_x = tl_x + image_width
    br_y = tl_y + image_height

    return tl_x, tl_y, br_x, br_y


def visualize_photos(tx, ty, photos, plot_size=1000, max_image_size=100):
    offset = max_image_size // 2
    image_centers_area_size = plot_size - 2 * offset

    tsne_plot = 255 * np.ones((plot_size, plot_size, 3), np.uint8)

    for x in range(len(photos)):
        image = cv2.imread(images_folder + '/' + photos[x])
        image = scale_image(image, max_image_size)
        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, tx[x], ty[x], image_centers_area_size, offset)
        tsne_plot[tl_y:br_y, tl_x:br_x, :] = image

    plt.imshow(tsne_plot[:, :, ::-1])
    plt.show()


if __name__ == '__main__':
    model = load_model()
    photos = get_all_photos_paths_from_file(images_folder)

    predictions = predict_with_features(model, photos)
    tsne = transform_to_tsne(predictions)

    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    visualize_photos(tx, ty, photos)
