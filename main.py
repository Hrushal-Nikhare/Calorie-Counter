# Imports
import os
import random
import json
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, flash, redirect, render_template, request, url_for
from tensorflow.keras.models import *
from tensorflow.keras.models import Model
from werkzeug.utils import secure_filename
import pickle

# Variables
UPLOAD_FOLDER = 'static\\uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'df0331cefc6c2b9a5d0208a726a5d1c0fd37324feba25506' # secret key for session
food_names = ['sutar_feni', 'sheera', 'sohan_papdi', 'sandesh', 'sohan_halwa', 'shrikhand', 'shankarpali', 'sheer_korma', 'unni_appam', 'ras_malai', 'pithe', 'paneer_butter_masala', 'pootharekulu', 'poornalu', 'rasgulla', 'rabri', 'poha', 'phirni', 'palak_paneer', 'qubani_ka_meetha', 'malapua', 'maach_jhol', 'navrattan_korma', 'modak', 'naan', 'misti_doi', 'lyangcha', 'makki_di_roti_sarson_da_saag', 'mysore_pak', 'misi_roti', 'kalakand', 'kadhi_pakoda', 'lassi', 'karela_bharta', 'kakinada_khaja', 'kajjikaya', 'kofta', 'ledikeni', 'litti_chokha', 'kuzhi_paniyaram', 'ghevar', 'imarti', 'dum_aloo', 'gulab_jamun', 'double_ka_meetha', 'kadai_paneer', 'kachori', 'gajar_ka_halwa', 'jalebi', 'gavvalu', 'doodhpak', 'chicken_tikka_masala', 'chicken_tikka', 'daal_puri', 'dal_makhani', 'chikki', 'daal_baati_churma', 'dharwad_pedha', 'chicken_razala', 'dal_tadka', 'chapati', 'chana_masala', 'boondi', 'bhatura', 'biryani', 'chhena_kheeri', 'butter_chicken', 'bhindi_masala', 'chak_hao_kheer', 'cham_cham', 'bandar_laddu', 'ariselu', 'aloo_gobi', 'aloo_tikki', 'aloo_shimla_mirch', 'anarsa', 'adhirasam', 'basundi', 'aloo_matar', 'aloo_methi']

# Functions
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_pairs_for_pred(image_path, images_):
	# initialize two empty lists to hold the (image, image) pairs and
	# labels to indicate if a pair is positive or negative
	random.seed(2023)
	pairImages = []
	pairLabels = []
   
	image = cv2.imread(image_path)
	resized_image = cv2.resize(image, (200,200))
				
	# loop over all images
	for idxA in range(len(images_)):
	
		currentImage = images_[idxA]

		pairImages.append([currentImage, resized_image]) 
   
	return (np.array(pairImages))

# Routes
@app.route('/', methods=['GET','POST'])
def upload_form():
	if request.method == 'POST':
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No image selected for uploading')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
			pairPred = create_pairs_for_pred(image_path, images_)
			image_prediction = model.predict( [pairPred[:,0],pairPred[:,1]])
			prediction = list(dish_dict.keys())[list(dish_dict.values()).index(np.argmax(image_prediction))]
			print(f"The image ({image_path}) is of {list(dish_dict.keys())[list(dish_dict.values()).index(np.argmax(image_prediction))]}")
			print('upload_image filename: ' + filename)
			return render_template('result.html', file=filename , prediction=prediction,calories=None,protein=None,fat=None,carbs=None,fact=None)
		else:
			flash('Allowed image types are -> png, jpg, jpeg, gif')
			return redirect(request.url)
	return render_template('index.html')

@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename=f'uploads/{filename}'), code=301)

@app.errorhandler(404)
def not_found(e):
	return render_template("404.html")

# Run Program
if __name__ == '__main__':
	# model = tf.keras.models.load_model('model')
	# model.summary()
	with open("images_.p","rb") as file:
		images_ = pickle.load(file)
	with open("labels.p","rb") as file:
		labels = pickle.load(file)
	with open("dish_dict.p","rb") as file:
		dish_dict = pickle.load(file)
	with open("data.json","r") as file:
		nutrition = json.load(file)
	for item in nutrition:
		for data_item in item['Food']:
			if data_item['Food'] == 'sutar_feni':
				print(data_item['Food'])
		
	#app.run(host='127.0.0.1', port=8000, debug=True)
