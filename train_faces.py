#Need to collect images first
import cv2
import os
from PIL import Image
import numpy as np
import tensorflow as tf
#PIL is the Python Image Library

import pickle
#Pickle is used to create new files

face_cascade = cv2.CascadeClassifier('/Python/Python37/cascades/data/haarcascade_frontalface_alt2.xml')

#The recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

#look for png or jpg files and adding to list
#Gives the current directory
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))

#Actual folder name where the images are stored
images_folder = "training_images"

#label will be the folder's name

#joining the images_folder path
image_dir = os.path.join(BASE_DIR, images_folder) 

current_id=0
label_ids = {}
x_train = []
y_labels = []

#Walking through the directory
for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			#joining the file
			path = os.path.join(root, file)

			#The folder with the images is the label for the training data
			#	It is the basename of the directory name
			#Can replace "os.path.dirname(path)" with just root because we are taking the basename of the root????
			label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
			print(label, path)

			#If the label is already accounted for, skip it
			if label in label_ids:
				pass
			else:
				#Using a dictionary to denote the ids of the labels
				#Adds the newest label to the dictionary and gives it the code which is the id
				label_ids[label] = current_id
				current_id += 1
			id_ = label_ids[label] #The actual number, as created above


			pil_image = Image.open(path).convert("L") #Grayscale conversion

			#Need to resize the images first, might want to normalize
			size = (550,550)
			final_image = pil_image.resize(size, Image.ANTIALIAS)

			image_array = np.array(final_image, "uint8") #Turning the image into an array of pixel-values
			#image_array = tf.cast(image_array, tf.float32)
			#np.divide(image_array, 255)

			print(image_array)

			#Face detection within the image
			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]

				#Add the training data
				x_train.append(roi)
				#Putting the id (of the label) because we need numbers for the actual labels which we are training with
				y_labels.append(id_)

#wd is writing bytes
#Label ids and their corresponding labels are now saved 
with open("labels.pkl", "wb") as f:
	pickle.dump(label_ids, f)

#Training the data with our training data  (y_labels only has numbers) (as numpy arrays)
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")
