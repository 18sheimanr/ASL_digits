from keras.models import Sequential
import wandb
from wandb.keras import WandbCallback
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split

#Sign language digits data set.
IMG_PATH = './ASL_X.npy'
LABEL_PATH = './ASL_Y.npy'
scaled_img_size = 32

wandb.init()
wandb.config.epochs = 18
wandb.config.batch_size = 16
wandb.config.hidden_layer_size = 128
wandb.config.conv_filters = 90

#return a dataframe of features and a list of corresponding labels
def load_data(img_path, label_path):
    x = np.load(img_path)
    y = np.load(label_path)
    return x, y

def buildModel():
    model = Sequential()
    model.add(Conv2D(wandb.config.conv_filters, (2, 2), input_shape=(scaled_img_size, scaled_img_size, 1)))  # input layer, conv+pool
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(wandb.config.conv_filters//2, (2, 2)))  #Convolution+Pooling again
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten()) #Flatten the 2D image arrays to flat 1D arrays
    model.add(Dense(wandb.config.hidden_layer_size, activation='relu')) #Hidden layer after convolution and pooling
    model.add(Dropout(0.3))
    model.add(Dense(wandb.config.hidden_layer_size//2, activation='relu'))  # Hidden layer after convolution and pooling
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='softmax'))  #output
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def transformImages(images):
    images = resize(images, (len(images), scaled_img_size, scaled_img_size))
    new_side_length = images.shape[1]
    images = images.astype('float32')
    return images.reshape(images.shape[0], new_side_length, new_side_length, 1)

#gather testing+training data and process it for the model

x, y = load_data(IMG_PATH, LABEL_PATH)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
x_train = transformImages(x_train)
x_test = transformImages(x_test)
labels = [9, 0, 7, 6, 1, 8, 4, 3, 2, 5]

model = buildModel()

model.fit(x_train, y_train, validation_data = x_test, epochs=wandb.config.epochs, batch_size=wandb.config.batch_size, callbacks=[WandbCallback(validation_data=x_test, labels=labels)])
#results automatically uploaded to wandb