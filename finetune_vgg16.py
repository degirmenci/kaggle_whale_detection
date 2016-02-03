from __future__ import absolute_import
from __future__ import print_function
#from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras.utils import np_utils, generic_utils
from six.moves import range
import scipy.io as scio
import h5py
import numpy

batch_size = 32 #CHANGE
nb_classes = 447
nb_epoch = 5


from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), stride=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), stride=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), stride=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), stride=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), stride=(2,2)))

    model.add(Flatten())
    model.add(Dense(512*7*7, 4096, activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(4096, 4096, activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(4096, 1000, activation='relu'))

    if weights_path:
        model.load_weights(weights_path)

    return model


# Test pretrained model
print('Loading the vgg16 model...')
model = VGG_16('vgg16_weights.h5')

model.add(Dropout(0.5))
model.add(Dense(1000, nb_classes, activation='softmax'))

# the data, shuffled and split between tran and test sets
print('Loading the dataset...')
compact_data = h5py.File('training_dataset_for_python_224x224_103015.mat')
X_train = compact_data['train_data'][()]
y_train = compact_data['train_label'][()]
X_test = compact_data['test_data'][()]
compact_data.close()

y_train = numpy.transpose(y_train,(1, 0))
X_train = numpy.transpose(X_train,(3, 2, 0, 1))
X_test = numpy.transpose(X_test,(3, 2, 0, 1))



print('X_train shape:', X_train.shape)
print(X_train.shape, 'train samples')
print(X_test.shape, 'test samples')
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)

# Split training part into training/validation
val_ratio = 0.1

print('Splitting training set into train and validation... Validation split ratio = ',val_ratio)
temp = numpy.random.permutation(X_train.shape[0])
temp2 = int(X_train.shape[0]*(1-val_ratio))

X_val = X_train[temp[temp2::],:,:,:]
X_train = X_train[temp[:temp2:],:,:,:]
Y_val = Y_train[temp[temp2::],:]
Y_train = Y_train[temp[:temp2:],:]




#sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
lr_curr = 0.005
#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

sgd = SGD(lr=lr_curr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

#X_train = X_train.astype("float32")
#X_test = X_test.astype("float32")
#X_val = X_val.astype("float32")
#X_train /= 255
#X_test /= 255
#X_val /= 255



data_augmentation = False
best_training_score = 1e23
best_val_score = 1e23

num_batches = int(numpy.ceil(float(X_train.shape[0])/batch_size))
if not data_augmentation:
    print("Not using data augmentation or normalization")

    for e in range(nb_epoch):
        progbar = generic_utils.Progbar(X_train.shape[0])
     
        for sub_e in range(num_batches):
            if (sub_e < (num_batches - 1)):
                X_batch = X_train[sub_e*batch_size:(sub_e+1)*batch_size,:,:,:]
                Y_batch = Y_train[sub_e*batch_size:(sub_e+1)*batch_size,:]
            else:
                X_batch = X_train[sub_e*batch_size::,:,:,:]
                Y_batch = Y_train[sub_e*batch_size::,:]
                
            loss = model.train_on_batch(X_batch, Y_batch)
            progbar.add(X_batch.shape[0], values=[("train loss", loss)])

        if (loss < best_training_score):
            print("Training error improved!")
            best_training_score = loss
        else:
            print("Training error got worse, decreasing learning rate...")
            lr_curr = lr_curr/5
            adam = Adam(lr=lr_curr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
            model.compile(loss='categorical_crossentropy', optimizer=adam)

        
           
        print("Validating...")
        # test time! 
        score = model.evaluate(X_val, Y_val, batch_size=1)
        print('Validation score:', score)
        if (score < best_val_score):
            best_val_score = score
            print('Best val. score so far! Saving the test predictions... epoch ', e)
else:
    print("Using real time data augmentation")

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.5,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.5,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    for e in range(nb_epoch):
        print('-'*40)
        print('Epoch', e)
        print('-'*40)
        print("Training...")
        # batch train with realtime data augmentation
        progbar = generic_utils.Progbar(X_train.shape[0])
        for X_batch, Y_batch in datagen.flow(X_train, Y_train):
            loss = model.train_on_batch(X_batch, Y_batch)
            progbar.add(X_batch.shape[0], values=[("train loss", loss)])

        if (loss < best_training_score):
            print("Training error improved!")
            best_training_score = loss
        else:
            print("Training error got worse, decreasing learning rate...")
            lr_curr = lr_curr/2
            adam = Adam(lr=lr_curr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
            model.compile(loss='categorical_crossentropy', optimizer=adam)
           
        print("Validating...")
        # test time!
        score = model.evaluate(X_val, Y_val, batch_size=1)
        print('Validation score:', score)
            
        if (score < best_val_score):
            best_val_score = score
            print('Best val. score so far! Saving the test predictions... epoch ', e)
            if (e > 450):
                prediction_1 = model.predict(X_test, batch_size = 1)
                temp_str = 'prediction_1_101515_v14_epoch_'+str(e)+'.mat'
                temp_str_2 = 'prediction_1_101515_v14_epoch_'+str(e)
                scio.savemat(temp_str, mdict = {temp_str_2: prediction_1})
            
                prediction_2 = model.predict(X_test_2, batch_size = 1)
                temp_str = 'prediction_2_101515_v14_epoch_'+str(e)+'.mat'
                temp_str_2 = 'prediction_2_101515_v14_epoch_'+str(e)
                scio.savemat(temp_str, mdict = {temp_str_2: prediction_2})

                prediction_3 = model.predict(X_test_3, batch_size = 1)
                temp_str = 'prediction_3_101515_v14_epoch_'+str(e)+'.mat'
                temp_str_2 = 'prediction_3_101515_v14_epoch_'+str(e)
                scio.savemat(temp_str, mdict = {temp_str_2: prediction_3})

                prediction_4 = model.predict(X_test_4, batch_size = 1)
                temp_str = 'prediction_4_101515_v14_epoch_'+str(e)+'.mat'
                temp_str_2 = 'prediction_4_101515_v14_epoch_'+str(e)
                scio.savemat(temp_str, mdict = {temp_str_2: prediction_4})
