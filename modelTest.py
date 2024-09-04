import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from keras.layers import InputLayer, Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy

dataset, dataset_info = tfds.load('malaria', with_info=True,
                                  as_supervised=True, 
                                  shuffle_files = True, 
                                  split=['train'])

def parasite_or_not(p):
    if p <= 0.5:
        return 'P'
    else:
        return 'U'
    
def resize_rescale(image, label):
    return tf.image.resize(image, (IM_SIZE, IM_SIZE))/255.0, label

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
IM_SIZE = 224


model = tf.keras.Sequential()
model.add(InputLayer(input_shape=(IM_SIZE, IM_SIZE,3)))
model.add(Conv2D(filters=6, kernel_size=5, strides=1, padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Conv2D(filters=16, kernel_size=5, strides=1, padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Flatten())

model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=Adam(learning_rate=0.01), loss=BinaryCrossentropy(), metrics=['acc'])
model.load_weights('./modelSaves/model.ckpt')

dataset = dataset[0]
junk = (dataset.take(int((TRAIN_RATIO+VAL_RATIO)*len(dataset))))
test_dataset = (dataset.skip(int((TRAIN_RATIO+VAL_RATIO)*len(dataset))))
test_dataset = test_dataset.map(resize_rescale)

test_dataset = test_dataset.batch(1)
model.evaluate(test_dataset)


    
for i, (image, label) in enumerate(test_dataset.take(9)):
    ax = plt.subplot(3, 3, i+1)
    plt.imshow(image[0])
    plt.title(str(parasite_or_not(label.numpy()[0])) + " : " + str(parasite_or_not(model.predict(image)[0][0])))

plt.axis('off')

plt.show()