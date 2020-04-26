"""
    This file loads all functionality of Xception pre-trained model.
"""
from tensorflow.keras.applications.xception import Xception

class Model:
    """
        This class is responsible for Loading of Xception Model.
        And loads all functionality of this model.
    """
    @classmethod
    def load_xception_model(cls):
        """
            This will initialize the model and returns the model.
        """
        return Xception(weights='imagenet', include_top=False, pooling='avg')

    @property
    def feature_dimentions(self):
        """
            This will return the feature dimensions.
        """
        return 2048
