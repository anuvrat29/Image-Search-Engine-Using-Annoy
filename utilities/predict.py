"""
    This file is responsible for predict the nearest images.
"""
# pylint: disable=E0611
# pylint: disable=E0001
# pylint: disable=W0702
# pylint: disable=E0401
import os
import numpy as np
from annoy import AnnoyIndex
from tensorflow.keras.preprocessing import image as IMAGE
from tensorflow.keras.applications.xception import preprocess_input
import pandas as pd

from utilities.model import Model
from utilities.config import MODEL_FOLDER

MODEL = Model().load_xception_model()

class Prediction:
    """
        This calss contains different methods to result similar images.
    """
    @classmethod
    def __init__(cls, model_name, number_of_images):
        """
            init method which will initialize this class.
        """
        cls.model_name = model_name
        cls.number_of_images = number_of_images

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
    def get_similar_item(cls, image_path, model, images):
        """
            This method will return the indexes of similar images.
            This method contains annoy indexer.
        """
        array = None
        n_image = cls.number_of_images
        try:
            image_data = cls.get_image_features(image_path)
            response = model.get_nns_by_vector(MODEL.predict(image_data).flatten(), n_image, include_distances=False)
            array = [images[response[index]] for index in range(0, n_image if n_image < len(response) else len(response))]
        except:
            array = ["-"] * n_image
        return array

    @classmethod
    def run_predict(cls, image_path):
        """
            This function will take care of finding nearest image.
        """
        m_name = cls.model_name
        annoy_details = None
        with open(os.path.join(os.path.join(MODEL_FOLDER, m_name), "config.txt"), "r") as file:
            annoy_details = file.read()
        annoy_details = annoy_details.split("\t")
        dataset = pd.read_csv(os.path.join(os.path.join(MODEL_FOLDER, m_name), m_name+".txt"), sep="\t")

        annoy_model = AnnoyIndex(int(annoy_details[0]), annoy_details[1])
        annoy_model.load(os.path.join(os.path.join(MODEL_FOLDER, m_name), m_name+".pkl"))
        array = cls.get_similar_item(image_path, annoy_model, list(dataset["Path"]))

        response = [os.path.basename(os.path.dirname(image))+"/"+os.path.basename(image) for image in array]
        return {"response": response}
