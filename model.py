import keras
import os
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import cv2
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D

# --------------------------Preporcess Data ----------------------------
data_dir = '/mnt/dcnas50/eng/testvn/data'
data = pd.read_csv(os.path.join(data_dir,'driving_log.csv'))
data['left_steering'] = data['steering'].apply(lambda x: x + .25 if np.abs(x) > .25 else x + .1)
data['right_steering'] = data['steering'].apply(lambda x: x - .25 if np.abs(x) > .25 else x - .1)

X_train_center = np.zeros((len(data),25,80,3))
X_train_left = np.zeros((len(data),25,80,3))
X_train_right = np.zeros((len(data),25,80,3))

for index,row in data.iterrows():
    filepath = os.path.join(data_dir,row['center'])
    img=mpimg.imread(filepath)
    img = img/255.0
    img = img[40:140,:]
    img = cv2.resize(img,(80,25))
    X_train_center[index,:,:,:] = img

    filepath = os.path.join(data_dir,row['left'].replace(' ', ''))
    img=mpimg.imread(filepath)
    img = img/255.0
    img = img[40:140,:]
    img = cv2.resize(img,(80,25))
    X_train_left[index,:,:,:] = img

    filepath = os.path.join(data_dir,row['right'].replace(' ', ''))
    img=mpimg.imread(filepath)
    img = img/255.0
    img = img[40:140,:]
    img = cv2.resize(img,(80,25))
    X_train_right[index,:,:,:] = img

y_train_center = np.array(data.steering)
y_train_left = np.array(data.left_steering)
y_train_right = np.array(data.right_steering)

X_train = np.append(X_train_center, X_train_left, axis = 0)
X_train = np.append(X_train, X_train_right, axis = 0)

y_train = np.append(y_train_center, y_train_left, axis = 0)
y_train = np.append(y_train, y_train_right, axis = 0)

X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size = .1,random_state = 7)

#------------------------------Define Model -------------------------------------

def final_model():
    # Best perfomance, fast, .0100 loss
    # create model
    model = Sequential()
    model.add(Convolution2D(60, 5, 5, border_mode='valid', input_shape=(25, 80, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(30, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, init = 'normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

#-----------------------------Training------------------------------------------
model = final_model()
early_stop_callback = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=1, verbose=1, mode='auto')]
model.fit(X_train, y_train, nb_epoch=15, batch_size=128, verbose=2, validation_data = (X_val,y_val), callbacks = early_stop_callback)

file_str = 'model'
model.save_weights(file_str+'.h5')

# save as JSON
json_string = model.to_json()

import json
with open(file_str+'.json', 'w') as outfile:
    json.dump(json_string, outfile)