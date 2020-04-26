"""
    Main controller which will communicate with frontend.
"""
import os
import glob
from flask import Flask, request, jsonify, send_file, after_this_request, render_template
from werkzeug.utils import secure_filename

from utilities.predict import Prediction
from utilities.traindatabase import TrainImageDatabase
from utilities.config import IMAGE_DATABASE, INPUT_IMAGE, MODEL_FOLDER

APP = Flask(__name__, template_folder=os.path.abspath("frontend"))

class ImageSearch:
    """
        This class contains all the methods and uris which will accept input and
        process those input and prepare the output.
    """
    @classmethod
    def save_files(cls, path, fileobject=None):
        """
            This method will save the images into mentioned path.
        """
        if not os.path.isdir(path):
            os.mkdir(path)
        else:
            for image in glob.glob(os.path.join(path, "*.*")):
                try:
                    os.remove(image)
                except:
                    pass
        if fileobject:
            for item in fileobject:
                filename = secure_filename(item.filename)
                item.save(os.path.join(path, filename))

    @APP.route("/", methods=["GET"])
    def index():
        """
            This will return index page of this usecase.
        """
        return render_template("index.html")

    @APP.route("/anuvrat/imageannoy/getmodels", methods=["GET"])
    def get_models():
        """
            this uri will calculate values of my images.
        """
        return jsonify({"models": next(os.walk(MODEL_FOLDER))[1]})

    @APP.route("/anuvrat/<modelname>/<imagename>", methods=["GET"])
    def get_images(modelname, imagename):
        """
            this uri will calculate values of my images.
        """
        return send_file(os.path.join(os.path.join(MODEL_FOLDER, modelname), imagename), as_attachment=False)

    @APP.route("/anuvrat/imageannoy/predict", methods=["POST"])
    def predict_image():
        """
            this uri will calculate values of my images.
        """
        inputimage = request.files.get("file")
        if not inputimage.filename:
            data = {"message": "Please upload a file & It seems without uploading file trying to Submit.",
                    "tag": "predict-div", "info": "display: none;"}
            return render_template("download.html", data=data)
        ImageSearch().save_files(INPUT_IMAGE)

        model_name = request.form.get("model")
        number_of_images = request.form.get("number_of_images")
        number_of_images = 5 if not number_of_images else number_of_images

        filename = inputimage.filename
        inputimage.save(os.path.join(INPUT_IMAGE, filename))
        data = Prediction(model_name, int(number_of_images)).run_predict(os.path.join(INPUT_IMAGE, filename))
        data["tag"] = "predict-div"
        data["info"] = "display: none;"
        return render_template("download.html", data=data)

    @APP.route("/anuvrat/imageannoy/prepare_model", methods=["POST"])
    def prepare_model():
        """
            this uri will calculate values of my images.
        """
        imagedatabase = request.files.getlist("files")
        if not imagedatabase:
            data = {"message": "Please upload a file & It seems without uploading file trying to Submit.",
                    "tag": "train-div", "info": "display: none;"}
            return render_template("download.html", data=data)
        ImageSearch().save_files(IMAGE_DATABASE, imagedatabase)

        metric = request.form.get("metric")
        annoy_tree = request.form.get("annoy_tree")
        annoy_tree = 15 if not annoy_tree else annoy_tree
        model_name = request.form.get("model_name")

        data = TrainImageDatabase(model_name).build_model(metric, annoy_tree)
        data["tag"] = "train-div"
        data["info"] = "display: visible;"
        return render_template("download.html", data=data)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    APP.run(port=65000, debug=True)
