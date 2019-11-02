import numpy as np
from keras import backend as K
from keras.optimizers import SGD,Adam,Adagrad
from keras.preprocessing import image as image_lib
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation, Conv2D, GlobalAveragePooling2D
from keras.layers import MaxPooling2D
from keras import applications
from keras import initializers
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
import imageio
import glob

#Create the dataset to be loaded in the model
src=r'C:\Users\aktorion\Desktop\Master\Master Ergasia\breast-histopathology-images\data1'+'\\'

# dimensions of our images.
img_size = 50

input_shape = (img_size, img_size, 3) #expected input size of our images

early_stopping = EarlyStopping(monitor='loss', patience=20) # stop after 20 worst results than best

#read images from files to matrices
def readImages(src,class_number,X_data=[],X_labels=[],start=0,end=-1,image_size = img_size):
    listOfFiles = os.listdir(src+'\\'+str(class_number)+'\\')
    if(end==-1):
        end=len(listOfFiles)
        
    for myFile in listOfFiles[start:end]:
        image = cv2.imread(src+'\\'+str(class_number)+'\\'+myFile).astype(np.float32)
        
        image = cv2.resize(image, (image_size, image_size), interpolation = cv2.INTER_CUBIC) 
        #center the luminosity

        image[:,:,0] -= 103.939
        image[:,:,1] -= 116.779
        image[:,:,2] -= 123.68
        image = image_lib.img_to_array(image)
        image.reshape((1,) + image.shape)
        X_data.append(image)
        X_labels.append(class_number)

def recall(y_true, y_pred):
     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
     recall = true_positives / (possible_positives + K.epsilon())
     return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

class TestCallback(Callback):
    def __init__(self, test_data_X, test_data_X_labels):
        self.test_data_X = test_data_X
        self.test_data_X_labels = test_data_X_labels
        

    def on_epoch_end(self, epoch, logs={}):
        x = self.test_data_X
        y = self.test_data_X_labels
        results = self.model.evaluate(x, y, verbose=0)
        loss, acc=results[0],results[1]
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

class WeightsSaver(Callback):
    def __init__(self, model, N=1):
        self.model = model
        self.N = N
        self.epoch = 1

    def on_epoch_end(self, epoch, logs={}):
        if self.epoch % self.N == 0:
            name = 'script1model1_50try17%05d.h5' % self.epoch
            self.model.save_weights(src + name)
        self.epoch += 1

#create the VGG 16 model
model = Sequential([
    Conv2D(64, (3, 3), input_shape=input_shape, padding='same',
           activation='relu'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(300,kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.01), bias_initializer='zeros'),
    Dense(300,kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.01), bias_initializer='zeros'),
    Dense(1,kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.01), bias_initializer='zeros')
])

# build the VGG16 network
base_model = applications.VGG16(include_top=False, weights='imagenet', input_shape=(img_size,img_size,3))

#Copy the values needed from initial model
for i in range(0,len(base_model.layers) - 1):
    (model.layers[i]).set_weights( (base_model.layers[i+1]).get_weights() )

#We train only the fully connected layer
for layer in model.layers[:-3]:
    layer.trainable = False
    
X_train_labels = []
X_train_data = []
readImages(src+'train',0,X_train_data,X_train_labels)
readImages(src+'train',1,X_train_data,X_train_labels)

X_validation_labels = []
X_validation_data = []
readImages(src+'validation',0,X_validation_data,X_validation_labels)
readImages(src+'validation',1,X_validation_data,X_validation_labels)

X_train_labels = np.asarray(X_train_labels)
X_train_data = np.asarray(X_train_data)

X_validation_labels = np.asarray(X_validation_labels)
X_validation_data = np.asarray(X_validation_data)


model.compile(optimizer=SGD(lr=0.00002, momentum=0.1, decay=0.000001, nesterov=True   ),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_data, X_train_labels,
          epochs=300,
          batch_size=150,
          #validation_data=(np.array(X_validation_data), np.array(X_validation_labels)),
          shuffle=True,
          callbacks=[WeightsSaver(model, 2),
                     early_stopping
                     ])
        