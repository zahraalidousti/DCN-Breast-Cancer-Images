import tensorflow as tf
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.optimizers import SGD, Adam 
from keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential

datagen = ImageDataGenerator(featurewise_center=True)

train_it = datagen.flow_from_directory('dataset/train',class_mode='binary', batch_size=8, target_size=(224, 224))

train_it.class_indices

train_it.classes

train_it.samples

test_it = datagen.flow_from_directory('dataset/test', class_mode='binary', batch_size=8, target_size=(224, 224))



# define cnn model
def VggNet_model():  
	# load model
  model = Sequential() 
  model.add(tf.keras.applications.VGG19(include_top=False, input_shape=(224, 224, 3), weights='imagenet'))
#mark loaded layers as not trainable
  for layer in model.layers:
    layer.trainable = False
# add new classifier layers

  model.add(Flatten())
  model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
  model.add(Dense(1, activation='sigmoid'))
	# compile model
  opt = SGD(lr=0.001, momentum=0.9)
  model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
  return model


# define cnn model
def ResNet_model():
	# load model
  model = Sequential()
  model.add(tf.keras.applications.ResNet152(include_top=False, input_shape=(224, 224, 3), weights='imagenet'))
# mark loaded layers as not trainable
  for layer in model.layers:
    layer.trainable = False
	# add new classifier layers
  model.add(Flatten())
  model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
  model.add(Dense(1, activation='sigmoid'))
	# compile model
  opt = SGD(lr=0.001, momentum=0.9)
  model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
  return model

def XceptionNet_model():
	# load model
  model = Sequential()
  model.add(tf.keras.applications.Xception(include_top=False, input_shape=(224, 224, 3), weights='imagenet'))
	# mark loaded layers as not trainable
  for layer in model.layers:
    layer.trainable = False
	# add new classifier layers
  model.add(Flatten())
  model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
  model.add(Dense(1, activation='sigmoid'))
	# compile model
  opt = SGD(lr=0.001, momentum=0.9)
  model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
  return model

def EfficientNetB_model():
	# load model
  model = Sequential()
  model.add(tf.keras.applications.EfficientNetB7(include_top=False, input_shape=(224, 224, 3), weights='imagenet'))
	# mark loaded layers as not trainable
  for layer in model.layers:
    layer.trainable = False
	# add new classifier layers
  model.add(Flatten())
  model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
  model.add(Dense(1, activation='sigmoid'))
	# compile model
  opt = SGD(lr=0.001, momentum=0.9)
  model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
  return model

def MobileNet_model():
	# load model
  model = Sequential()
  model.add(tf.keras.applications.MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights='imagenet'))
	# mark loaded layers as not trainable
  for layer in model.layers:
    layer.trainable = False
	# add new classifier layers
  model.add(Flatten())
  model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
  model.add(Dense(1, activation='sigmoid'))
	# compile model
  opt = SGD(lr=0.001, momentum=0.9)
  model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
  return model

def DenseNet_model():
	# load model
  model = Sequential()
  model.add(tf.keras.applications.DenseNet201(include_top=False, input_shape=(224, 224, 3), weights='imagenet'))
	# mark loaded layers as not trainable
  for layer in model.layers:
    layer.trainable = False
	# add new classifier layers
  model.add(Flatten())
  model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
  model.add(Dense(1, activation='sigmoid'))
	# compile model
  opt = SGD(lr=0.001, momentum=0.9)
  model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
  return model

def InceptionNet_model():
	# load model
  model = Sequential()
  model.add(tf.keras.applications.InceptionResNetV2(include_top=False, input_shape=(224, 224, 3), weights='imagenet'))
	# mark loaded layers as not trainable
  for layer in model.layers:
    layer.trainable = False
	# add new classifier layers
  model.add(Flatten())
  model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
  model.add(Dense(1, activation='sigmoid'))
	# compile model
  opt = SGD(lr=0.001, momentum=0.9)
  model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
  return model


# model = VggNet_model()
# model = ResNet_model()
# model = XceptionNet_model()
# model = EfficientNetB_model()
model = MobileNet_model()
# model = DenseNet_model()
# model = InceptionNet_model()

history = model.fit(train_it, steps_per_epoch=len(train_it), epochs=50, verbose=1, validation_data=test_it)


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=1)
	print('Accuracy: > %.3f' % (acc * 100.0))

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(224, 224))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 224, 224, 3) #add a new layer
	# center pixel data
	img = img.astype('float32')
	return img

# load an image and predict the class
def run_example(model):
	# img = load_image('dataset/test/Sick/269-R-front.jpg')
 	img = load_image('dataset/test/Healthy/93-L-front.jpg')

 	result = model.predict(img)
 	if result < 0.5:

	  print(result[0], 'Healthy')
 	else:
	  print(result[0], 'Sick')

run_example(model)

pwd

import os

a=0
b=0
c=0
d=0

def CM_healthy(i, path,a,b):
 	img = load_image(os.path.join(path,i))
 	result = model.predict(img)
 	if result < 0.5:
 	   a = a+1
 	else:
 	   b = b+1
 	return a,b

def CM_sick(i, path,c,d):
 	img = load_image(os.path.join(path,i))
 	result = model.predict(img)
 	if result < 0.5:
 	   d = d+1
 	else:
 	   c = c+1
 	return c,d


path_Healthy = 'dataset/test/Healthy/'
path_Sick = 'dataset/test/Sick/'

for img in os.listdir(path_Healthy):
  a,b = CM_healthy(img, path_Healthy,a,b)

for img in os.listdir(path_Sick):
  c,d = CM_sick(img, path_Sick,c,d)

import pandas as pd
confusion_matrix = [[a,b],[c,d]]
print('confusion_matrix of model \n')
pd.DataFrame(confusion_matrix, columns = ['healthy', 'sick'], index=['healthy', 'sick'])
