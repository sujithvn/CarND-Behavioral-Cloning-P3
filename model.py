import csv
from scipy import ndimage
from random import shuffle
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

print("Started...")
# Read the Driving log which is available in a CSV format
# We are interested in the first four columns
# The first three columns has the path to Center, Left & Right camera image
# The fourth column is the steering angle [this will be our label column)
# contents are saved to a list
lines = []
with open('./tempdata/driving_log_norm.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        lines.append(row)
print("Drive log read, total rows: ", len(lines))

# Split the input into Train & Test data in a 80:20 ratio
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# Defining a generator function. This will take the input files in batches of size 'batch_size 32'
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while(1):
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            correction = 0.2
            # We are correcting the steering angle by +0.2 for left camera image
            # and a correction of -0.2 for the right camera image
            # The images and the corresponding steering angles are collected

            for batch_sample in batch_samples:
                for i in range(3):    
                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    cur_path = './tempdata/IMG/' + filename
                    # cur_path = source_path # Use the same path if training locally
                    image = ndimage.imread(cur_path)
                    images.append(image)
                    if i == 0:
                        measurements.append(float(batch_sample[3]))
                    elif i == 1: # left camera
                        measurements.append(float(batch_sample[3]) + correction)
                    else:        # right camera
                        measurements.append(float(batch_sample[3]) - correction)


            # The images were collected while going through the track in a anti-clockwise for 2 loops
            # This will give a tendency to predict LEFT more often. 
            # To fix this, we need training data with right turns (later we added 1 loop of clockwise driving)
            # Alternate option used here instead of collecting more data is to FLIP the images horizontally
            # This gives a perspective of turning right. The corresponding steering angle is negated
            aug_images, aug_measus = [], []
            for image, measu in zip(images, measurements):
                aug_images.append(image)
                aug_measus.append(measu)
                aug_images.append(np.fliplr(image)) #  cv2.flip(image, 1)
                aug_measus.append(-1.0 * measu)

            # Training data is ready for this batch and sent back by the generator using the YIELD function
            X_train = np.array(images)
            y_train = np.array(measurements)

            yield sklearn.utils.shuffle(X_train, y_train)


# Calling the generator function to get data processed in batches.
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

print("Start modelling...")
# Model arch used initially was based on the standard LeNet architecture.
# It was changed to the NVIDIA e2e architecture as implemented below for better performance.
# We added two pre-processing steps to Normalize the image and CROP of the unwanted section of the image
model = Sequential()

# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

model.add(Cropping2D(((70, 25), (0, 0))))

#1st layer - Conv2D, no of filters- 24, filter size= 5x5, stride= 2x2
model.add(Convolution2D(24, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Dropout(0.1))

#2nd layer - Conv2D, no of filters- 36, filter size= 5x5, stride= 2x2
model.add(Convolution2D(36, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Dropout(0.2))

#3rd layer - Conv2D, no of filters- 48, filter size= 5x5, stride= 2x2
model.add(Convolution2D(48, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Dropout(0.1))

#4th layer - Conv2D, no of filters- 64, filter size= 3x3, stride= 1x1
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.2))

#5th layer - Conv2D, no of filters- 64, filter size= 3x3, stride= 1x1
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.1))

#flatten
model.add(Flatten())

#6th layer - fully connected layer
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))

#7th layer - fully connected layer
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))

#8th layer - fully connected layer
model.add(Dense(10, activation='relu'))

#9th layer - fully connected layer
model.add(Dense(1)) # Final prediction of the steering angle.

print("Model compiling...")
# The accuracy metrics used is MSE (Mean Squared Error) since we are predicting a regression value.
# The optimizer is 'ADAM'
model.compile(loss='mse', optimizer='adam')
print("Model fitting...")
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5, verbose = 1)
model.save('model.h5')
print("Model saved above and exiting...")
exit()