"""
    This file is responsible for training or creating indexes on images
    and return similar indexes depending upon ranks.
"""
# pylint: disable=E0611
# pylint: disable=E0001
# pylint: disable=W0702
import os
import glob
import datetime
import numpy as np
from annoy import AnnoyIndex
from tensorflow.keras.preprocessing import image as IMAGE
from tensorflow.keras.applications.xception import preprocess_input

from utilities.model import Model
from utilities.config import MODEL_FOLDER, IMAGE_DATABASE

MODEL = Model().load_xception_model()

class TrainImageDatabase:
    """
        This calss contains different methods to result similar images.
    """
    @classmethod
    def __init__(cls, name):
        """
            init method which will initialize this class.
        """
        cls.name = name

    @classmethod
    def get_image_features(cls, image_path):
        """
            This method will return the image features.
        """
        image = IMAGE.load_img(image_path, target_size=(224, 224))
        image_data = IMAGE.img_to_array(image)
        image_data = np.expand_dims(image_data, axis=0)
        image_data = preprocess_input(image_data)
        return image_data

    @classmethod
    def build_annoy_tree(cls, features, metric, folder_name, annoy_tree):
        """
            This mehod will build the trees uses Annoy algorithm for the same.
            annoy_index.build(x) if x is larger then it will increase accuracy
            but be aware the overfitting problem. x is number of trees.
        """
        annoy_index = AnnoyIndex(Model().feature_dimentions, metric)
        with open(os.path.join(os.path.join(MODEL_FOLDER, folder_name), "config.txt"), "w") as file:
            file.write(f"{Model().feature_dimentions}\t{metric}")
        for key, value in features.items():
            annoy_index.add_item(key, MODEL.predict(value).flatten())
        annoy_index.build(annoy_tree)
        information = {"feature_dimension": Model().feature_dimentions, "metric": metric}
        return annoy_index, information

    @classmethod
    def model_folder_name(cls):
        """
            This method will return model file name with extension
            and folder name where all the data will be stored.
        """
        if not os.path.isdir(MODEL_FOLDER):
            os.mkdir(MODEL_FOLDER)
        date = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
        folder_name = date if not cls.name else cls.name
        model_name = folder_name + ".pkl"
        if not os.path.isdir(os.path.join(MODEL_FOLDER, folder_name)):
            os.mkdir(os.path.join(MODEL_FOLDER, folder_name))
        return folder_name, model_name

    @classmethod
    def build_model(cls, metric, annoy_tree):
        """
            This method will calculate the feature of images
        """
        features, image_path = {}, {}
        for index, path in enumerate(glob.glob(os.path.join(IMAGE_DATABASE, "*.*"))):
            try:
                features[index] = cls.get_image_features(path)
                image_path[index] = path
            except:
                pass

        folder_name, model_name = cls.model_folder_name()
        annoy_model, information = cls.build_annoy_tree(features, metric, folder_name, annoy_tree)
        information["model_name"] = model_name

        with open(os.path.join(os.path.join(MODEL_FOLDER, folder_name), f"{folder_name + '.txt'}"), "w") as file:
            file.write("Index\tPath\n")
            for key, value in image_path.items():
                path = os.path.join(os.path.join(MODEL_FOLDER, folder_name), os.path.basename(value))
                os.rename(value, path)
                file.write(f"{key}\t{path}\n")

        annoy_model.save(os.path.join(os.path.join(MODEL_FOLDER, folder_name), model_name))
        information["message"] = "Model Saved Successfully at server. Please go for prediction."
        return information
