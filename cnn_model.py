# Part 1 - Membuat Model CNN

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#Import library Keras dan paketnya
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers

# Inisialisasi Model CNN
classifier = Sequential()

# Langkah 1 - Convolution Layer
classifier.add(Convolution2D(32, (3, 3), activation = 'relu', input_shape = (64, 64, 3)))

# Langkah 2 - Pooling
classifier.add(MaxPooling2D(pool_size =(2,2)))

# Langkah 3 - 2nd Convolution Layer
classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))

# Langkah 4 - 3rd Convolution Layer
classifier.add(Convolution2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))

# Langkah 5 - Flattening
classifier.add(Flatten())

# Langkah 6 - Full Connection
classifier.add(Dense(256, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(41, activation = 'softmax'))

# Compile Model CNN
classifier.compile(optimizer = optimizers.SGD(learning_rate = 0.01), loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.summary()


# Part 2 - Fittting the CNN to the Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('mydata/training_set', target_size=(64, 64), batch_size=32, class_mode='categorical')

test_set = test_datagen.flow_from_directory('mydata/test_set', target_size=(64, 64), batch_size=32, class_mode='categorical')

model = classifier.fit(training_set, steps_per_epoch=800, epochs=27, validation_data = test_set, validation_steps = len(test_set))
#model = classifier.fit(training_set, steps_per_epoch=len(training_set), epochs=30, validation_data = test_set, validation_steps = len(test_set))

"""#Saving the model
import h5py
classifier.save('Trained_model.h5')"""

print(model.history.keys())
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(model.history['accuracy'], label='train')
plt.plot(model.history['val_accuracy'], label='test')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()
# summarize history for loss

plt.plot(model.history['loss'], label='train')
plt.plot(model.history['val_loss'], label='test')
plt.title('Model Loss')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()








