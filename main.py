# Imports
import os
import random

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, flash, redirect, render_template, request, url_for
from tensorflow.keras.models import *
from tensorflow.keras.models import Model
from werkzeug.utils import secure_filename

# Variables
UPLOAD_FOLDER = 'C:\\Users\\hrush\\OneDrive\\Desktop\\Programs\\School\\Food Calorie Finder\\Draft 3\\static\\uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'df0331cefc6c2b9a5d0208a726a5d1c0fd37324feba25506' # secret key for session
model_path = 'C:\\Users\\hrush\\OneDrive\\Desktop\\Programs\\School\\Food Calorie Finder\\Draft 3\\static\\model.h5'
food_names = ['sutar_feni', 'sheera', 'sohan_papdi', 'sandesh', 'sohan_halwa', 'shrikhand', 'shankarpali', 'sheer_korma', 'unni_appam', 'ras_malai', 'pithe', 'paneer_butter_masala', 'pootharekulu', 'poornalu', 'rasgulla', 'rabri', 'poha', 'phirni', 'palak_paneer', 'qubani_ka_meetha', 'malapua', 'maach_jhol', 'navrattan_korma', 'modak', 'naan', 'misti_doi', 'lyangcha', 'makki_di_roti_sarson_da_saag', 'mysore_pak', 'misi_roti', 'kalakand', 'kadhi_pakoda', 'lassi', 'karela_bharta', 'kakinada_khaja', 'kajjikaya', 'kofta', 'ledikeni', 'litti_chokha', 'kuzhi_paniyaram', 'ghevar', 'imarti', 'dum_aloo', 'gulab_jamun', 'double_ka_meetha', 'kadai_paneer', 'kachori', 'gajar_ka_halwa', 'jalebi', 'gavvalu', 'doodhpak', 'chicken_tikka_masala', 'chicken_tikka', 'daal_puri', 'dal_makhani', 'chikki', 'daal_baati_churma', 'dharwad_pedha', 'chicken_razala', 'dal_tadka', 'chapati', 'chana_masala', 'boondi', 'bhatura', 'biryani', 'chhena_kheeri', 'butter_chicken', 'bhindi_masala', 'chak_hao_kheer', 'cham_cham', 'bandar_laddu', 'ariselu', 'aloo_gobi', 'aloo_tikki', 'aloo_shimla_mirch', 'anarsa', 'adhirasam', 'basundi', 'aloo_matar', 'aloo_methi']

# Functions
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def loadimgs_(names):
	'''
	path => Path of train directory or test directory
	'''
	images = []
	labels = [] 

	img_shape = (200,200)
	dish_dict = {}

	path = "C:\\Users\\hrush\\Downloads\\archive\\Indian Food Images\\Indian Food Images"

	category_images = []

	for clsctr, dish in enumerate(names):
		dish_dict[dish] = clsctr
		dish_path = f'{os.path.join(path, dish)}/'

		for count, filename in enumerate(os.listdir(dish_path)):
						#print("Class :{}, count: {}".format(clsctr,count))
			if count == 1:
				#print("In train condition")
				image_path = os.path.join(dish_path, filename)
				image = cv2.imread(image_path)
				resized_image = cv2.resize(image, img_shape)
				final_image = resized_image #cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
				labels.append(clsctr)
				try:
					category_images.append(final_image)

				except ValueError as e:
					print(e)
					print("error - category_images_train:", final_image)
	images = category_images

	return images, labels, dish_dict


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
			print(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			#print('upload_image filename: ' + filename)
			flash('Image successfully uploaded and displayed below')
			return render_template('result.html', file=filename)
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
	# model = tf.keras.models.load_model('C:\\Users\\hrush\\OneDrive\\Desktop\\Programs\\School\\Food Calorie Finder\\Draft 3\\model')
	# model.summary()
	# image_path ="C:\\Users\\hrush\\Downloads\\archive\\Indian Food Images\\Indian Food Images\\gajar_ka_halwa\\0b936843f0.jpg"
	# images_, labels_ , dish_dict_ = loadimgs_(food_names) 
	# pairPred = create_pairs_for_pred(image_path, images_)
	# image_prediction = model.predict( [pairPred[:,0],pairPred[:,1]])
	# print(f"The image is of {list(dish_dict.keys())[list(dish_dict.values()).index(np.argmax(image_prediction))]}")
	app.run(host='127.0.0.1', port=8000, debug=True)
