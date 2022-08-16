# Imports
import os # file saving
import random # random number generator
import json # reading json file
import cv2 # image processing
import numpy as np # numpy
import tensorflow as tf # tensorflow
from flask import Flask, flash, redirect, render_template, request, url_for
from tensorflow.keras.models import *
from tensorflow.keras.models import Model
from werkzeug.utils import secure_filename #file name check
import pickle # read files

# Variables
UPLOAD_FOLDER = 'static\\uploads' # upload folder to store files in
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'} # allowed file types
app = Flask(__name__) # init flask app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER # assign upload folder
app.config['SECRET_KEY'] = 'df0331cefc6c2b9a5d0208a726a5d1c0fd37324feba25506' # secret key for session
food_names = ['sutar_feni', 'sheera', 'sohan_papdi', 'sandesh', 'sohan_halwa', 'shrikhand', 'shankarpali', 'sheer_korma', 'unni_appam', 'ras_malai', 'pithe', 'paneer_butter_masala', 'pootharekulu', 'poornalu', 'rasgulla', 'rabri', 'poha', 'phirni', 'palak_paneer', 'qubani_ka_meetha', 'malapua', 'maach_jhol', 'navrattan_korma', 'modak', 'naan', 'misti_doi', 'lyangcha', 'makki_di_roti_sarson_da_saag', 'mysore_pak', 'misi_roti', 'kalakand', 'kadhi_pakoda', 'lassi', 'karela_bharta', 'kakinada_khaja', 'kajjikaya', 'kofta', 'ledikeni', 'litti_chokha', 'kuzhi_paniyaram', 'ghevar', 'imarti', 'dum_aloo', 'gulab_jamun', 'double_ka_meetha', 'kadai_paneer', 'kachori', 'gajar_ka_halwa', 'jalebi', 'gavvalu', 'doodhpak', 'chicken_tikka_masala', 'chicken_tikka', 'daal_puri', 'dal_makhani', 'chikki', 'daal_baati_churma', 'dharwad_pedha', 'chicken_razala', 'dal_tadka', 'chapati', 'chana_masala', 'boondi', 'bhatura', 'biryani', 'chhena_kheeri', 'butter_chicken', 'bhindi_masala', 'chak_hao_kheer', 'cham_cham', 'bandar_laddu', 'ariselu', 'aloo_gobi', 'aloo_tikki', 'aloo_shimla_mirch', 'anarsa', 'adhirasam', 'basundi', 'aloo_matar', 'aloo_methi']
# Food names the AI will predict from ^^
# Functions
def allowed_file(filename): # check if file is allowed
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_pairs_for_pred(image_path, images_):
	# initialize two empty lists to hold the (image, image) pairs and
	# labels to indicate if a pair is positive or negative
	random.seed(2023) # seed random number generator
	pairImages = []
	pairLabels = []
   
	image = cv2.imread(image_path) # read image
	resized_image = cv2.resize(image, (200,200)) # resize image
				
	# loop over all images
	for idxA in range(len(images_)): # loop over all images
	
		currentImage = images_[idxA] # get current image

		pairImages.append([currentImage, resized_image])  # append image to pairs
   
	return (np.array(pairImages)) # return pairs

# Routes
@app.route('/', methods=['GET','POST']) # home page
def upload_form():
	if request.method == 'POST': # if post
		if 'file' not in request.files:
			flash('No file part') # flash error
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '': # if no file selected
			flash('No image selected for uploading')
			return redirect(request.url)
		if file and allowed_file(file.filename): # if file is allowed
			filename = secure_filename(file.filename) # secure filename check
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename)) # save file
			image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename) # get image path
			pairPred = create_pairs_for_pred(image_path, images_) # create pairs for prediction
			image_prediction = model.predict( [pairPred[:,0],pairPred[:,1]]) # predict image
			prediction = list(dish_dict.keys())[list(dish_dict.values()).index(np.argmax(image_prediction))] # get prediction
			# output prediction
			return render_template('result.html', file=filename , prediction=prediction,calories=nutrition[dish_dict[prediction]]['CALORIES'],protein=nutrition[dish_dict[prediction]]['PROTEIN'],fat=nutrition[dish_dict[prediction]]['FATS'],carbs=nutrition[dish_dict[prediction]]['CARBS'])
		else:
			flash('Allowed image types are -> png, jpg, jpeg, gif') # flash error
			return redirect(request.url)
	return render_template('index.html') # return index

@app.route('/display/<filename>') # display image
def display_image(filename):
	return redirect(url_for('static', filename=f'uploads/{filename}'), code=301) # redirect to image

@app.errorhandler(404) # 404 error
def not_found(e):
	return render_template("404.html") # return 404

# Run Program
if __name__ == '__main__':
	model = tf.keras.models.load_model('model') # load model
	model.summary() # print model summary
	with open("images_.p","rb") as file:
		images_ = pickle.load(file) # load images_ variable
	with open("labels.p","rb") as file:
		labels = pickle.load(file) # load labels variable
	with open("dish_dict.p","rb") as file:
		dish_dict = pickle.load(file) # load dish_dict variable
	with open("data.json","r") as file:
		nutrition = json.load(file) # load nutrition variable

	app.run(host='127.0.0.1', port=8000, debug=True) # run app on localhost:8000
