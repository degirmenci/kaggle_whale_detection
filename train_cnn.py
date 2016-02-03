from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras.utils import np_utils, generic_utils
from six.moves import range
import scipy.io as scio
import numpy

batch_size = 32 #CHANGE
nb_classes = 447
nb_epoch = 15

# the data, shuffled and split between tran and test sets
compact_data = scio.loadmat('training_dataset_for_python_v1_100815.mat')
X_train = compact_data['train_data']
y_train = compact_data['train_label']
X_test = compact_data['test_data']

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


print('Creating the CNN model...')
model = Sequential()

# Initial arguments
nr_kernels1 = 32 # arbitrary choice
kernel_size_1 = 7  # arbitrary choice
kernel_size_2 = 5  # arbitrary choice
kernel_size_3 = 3  # arbitrary choice
nr_channels = X_train.shape[1]  # RGB inputs => 3 channels

#1
# model.add(Convolution2D(32, 3, 3, 3, border_mode='full'))

model.add(Convolution2D(nr_kernels1,
                        nr_channels,
                        kernel_size_1,
                        kernel_size_1,
                        border_mode='full'))

model.add(Activation('relu'))
# (b x 3 x 32 x 32) -> (b x 32 x 32 x 32)


nr_kernels2 = 32 #arbitrary choice
model.add(Convolution2D(
        nr_kernels2,  # I chose it
        nr_kernels1,  # derived from output of previous layer
        kernel_size_2,  # I chose it
        kernel_size_2)) # I chose it

model.add(Activation('relu'))


#3
model.add(MaxPooling2D(poolsize=(2, 2))) # arbitrary choice
model.add(Dropout(0.25))
# (b x 32 x 32 x 32) -> (b x 32 x 16 x 16)


#4
nr_kernels3=32
model.add(Convolution2D(nr_kernels3, # I chose it
                        nr_kernels2,
                        kernel_size_3,
                        kernel_size_3,
                        border_mode='full'))
model.add(Activation('relu'))
# (b x 32 x 16 x 16) -> (b x 64 x 16 x 16)


#5
nr_kernels4=32
model.add(Convolution2D(nr_kernels4,
                        nr_kernels3,
                        kernel_size_3,
                        kernel_size_3))
model.add(Activation('relu'))
# (b x 64 x 16 x 16) -> (b x 64 x 16 x 16)


#6
model.add(MaxPooling2D(poolsize=(2, 2))) # arbitrary
model.add(Dropout(0.25))
# (b x 64 x 16 x 16) -> (b x 64 x 8 x 8)


#7
model.add(Flatten())


#8
# The dimensions of each image is (N = (X_train.shape[2]) ** 2),
# We had two downsampling layers of 2x2 maxpooling, so we divide each dimension twice by 2 (/2 /2).
# The input to this layer is the 64 "channels" that the previous layer outputs. Thus we have a layer of
# nr_kernels * (N / 2 / 2) * (N / 2 / 2)

flat_layer_size = nr_kernels4 * (X_train.shape[2] / 2 / 2) ** 2
final_layer_size=512 # I chose it
interm_layer_size = 512 # I chose it
model.add(Dense(flat_layer_size, interm_layer_size))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(interm_layer_size, final_layer_size))
model.add(Activation('relu'))
model.add(Dropout(0.5))


#9
model.add(Dense(final_layer_size, nb_classes))
model.add(Activation('softmax'))

#sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model.compile(loss='categorical_crossentropy', optimizer=adam)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_val = X_val.astype("float32")
X_train /= 255
X_test /= 255
X_val /= 255



data_augmentation = True

if not data_augmentation:
    print("Not using data augmentation or normalization")


    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch)
    score = model.evaluate(X_test, Y_test, batch_size=batch_size)
    print('Test score:', score)

else:
    print("Using real time data augmentation")

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=360,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
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

        print("Testing...")
        # test time! 
        score = model.evaluate(X_val, Y_val, batch_size=1)
        print('Validation score:', score)

        
    print('Saving the test predictions... epoch ', e)
    prediction = model.predict(X_test, batch_size = 1)
    temp_str = 'prediction_101115_v8p3_epoch_'+str(e)+'.mat'
    temp_str_2 = 'prediction_101115_v8p3_epoch_'+str(e)
    scio.savemat(temp_str, mdict = {temp_str_2: prediction})


