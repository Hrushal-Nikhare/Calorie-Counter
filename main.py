# Imports
import os
import random
import json
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, flash, redirect, render_template, request, url_for
import webbrowser
from werkzeug.utils import secure_filename
import pickle

# Variables
UPLOAD_FOLDER = "static\\uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Functions
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def create_pairs_for_pred(image_path, images_):
    """Create pairs for prediction."""
    random.seed(2023)  # seed random number generator
    pairImages = []

    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (200, 200))  # resize image
    # loop over all images and get current image
    for idxA in range(len(images_)):
        currentImage = images_[idxA]
        pairImages.append([currentImage, resized_image])  # append image to pairs

    return np.array(pairImages)  # return pairs


# Routes
@app.route("/", methods=["GET", "POST"])  # home page
def upload_form():
    """Home page."""
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")  # flash error
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":  # if no file selected
            flash("No image selected for uploading")
            return redirect(request.url)
        if file and allowed_file(file.filename):  # if file is allowed
            filename = secure_filename(file.filename)  # secure filename check
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))  # save file
            image_path = os.path.join(
                app.config["UPLOAD_FOLDER"], filename
            )  # get image path
            pairPred = create_pairs_for_pred(
                image_path, images_
            )  # create pairs for prediction
            image_prediction = model.predict(
                [pairPred[:, 0], pairPred[:, 1]]
            )  # predict image
            prediction = list(dish_dict.keys())[
                list(dish_dict.values()).index(np.argmax(image_prediction))
            ]  # get prediction
            # output prediction
            return render_template(
                "result.html",
                file=filename,
                prediction=prediction.replace("_", " "),
                calories=nutrition[dish_dict[prediction]]["CALORIES"],
                protein=nutrition[dish_dict[prediction]]["PROTEIN"],
                fat=nutrition[dish_dict[prediction]]["FATS"],
                carbs=nutrition[dish_dict[prediction]]["CARBS"],
            )
        else:
            flash("Allowed image types are -> png, jpg, jpeg, gif")  # flash error
            return redirect(request.url)
    return render_template("index.html")  # return index


@app.route("/display/<filename>")  # display image
def display_image(filename):
    """display image"""
    return redirect(url_for("static", filename=f"uploads/{filename}"), code=301)  # redirect to image


@app.errorhandler(404)  # 404 error
def not_found():
    """Page not found."""
    return render_template("404.html")  # return 404


# Run Program
if __name__ == "__main__":
    model = tf.keras.models.load_model("model")  # load model
    # model.summary() # print model summary / dont need
    with open("images_.p", "rb") as file:
        images_ = pickle.load(file)  # load images_ variable
    with open("labels.p", "rb") as file:
        labels = pickle.load(file)  # load labels variable
    with open("dish_dict.p", "rb") as file:
        dish_dict = pickle.load(file)  # load dish_dict variable
    with open("data.json", "r", encoding="utf-8") as file:
        nutrition = json.load(file)  # load nutrition variable

    from waitress import serve

    webbrowser.open("http://localhost:8080")
    print("\nServer is running...")
    print("Open http://localhost:8080 to view it in the browser.")
    print("Press Ctrl + C to stop the server.\n")

    serve(app, host="127.0.0.1", port=8080)

    # app.run(host="127.0.0.1", port=8000)  #debug
