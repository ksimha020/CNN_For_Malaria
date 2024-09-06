import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from typing import List, Union
import tensorflow as tf
import tensorflow_datasets as tfds


# Subclassed Feature extraction
class FeatureExtraction(tf.keras.layers.Layer):
    def __init__(self, filterList: List[int], kernelSize: List[int], strides: List[int], padding: List[str], activation: List[str], poolSize: List[int]):
        if not self.same_length(filterList, kernelSize, strides, padding, activation, poolSize):
            raise ValueError('Lengths of inputted lists is unequal')
        super(FeatureExtraction, self).__init__()
        self.convLayers = []
        self.batchLayers = []
        self.poolingLayers = []

        for filterNum, kernelNum, stride, pad, acti, pool in zip(filterList, kernelSize, strides, padding, activation, poolSize):
            self.convLayers.append(tf.keras.layers.Conv2D(filters=filterNum, kernel_size=kernelNum, strides=stride, padding=pad, activation=acti))
            self.batchLayers.append(tf.keras.layers.BatchNormalization())
            self.poolingLayers.append(tf.keras.layers.MaxPool2D(pool_size=pool, strides=stride*2))

    def call(self, x, training=False):
        for conv, batch, pool in zip(self.convLayers, self.batchLayers, self.poolingLayers):
            x = conv(x)
            x = batch(x)
            x = pool(x)
        return x

    @staticmethod
    def same_length(*lists: List[Union[str, int]]) -> bool:
        return len(set([len(lst) for lst in lists])) == 1
    
# Custom class for dense networks towards tail end of neural network
class DenseNetwork(tf.keras.layers.Layer):
    def __init__(self, denseList: List[int], activation: List[str]):
        if not self.same_length(denseList, activation):
            raise ValueError('Lengths of inputted lists is unequal')
        super(DenseNetwork, self).__init__()
        self.network = []
        self.batches = []
        for denseNum, acti in zip(denseList[:-1], activation[:-1]):
            self.network.append(tf.keras.layers.Dense(denseNum, activation=acti))
            self.batches.append(tf.keras.layers.BatchNormalization())
        self.lastDense = tf.keras.layers.Dense(denseList[-1], activation=activation[-1])

    def call(self, x, training=False):
        for denseNet, batches in zip(self.network, self.batches):
            x = denseNet(x)
            x = batches(x)
        x = self.lastDense(x)
        return x
    
    @staticmethod
    def same_length(*lists: List[Union[str, int]]) -> bool:
        return len(set([len(lst) for lst in lists])) == 1
    
class CustomCNNModel(tf.keras.models.Model):
    def __init__(self, filterList: List[int], kernelSize: List[int], strides: List[int], padding: List[str], activationCNN: List[str], poolSize: List[int], denseList: List[int], activationDense: List[str]):
        super(CustomCNNModel, self).__init__()
        self.feature_extractor = FeatureExtraction(filterList, kernelSize, strides, padding, activationCNN, poolSize)
        self.flatten = tf.keras.layers.Flatten()
        self.network = DenseNetwork(denseList, activationDense)

    def call(self, x, training=False):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = self.network(x)
        return x


dataset, dataset_info = tfds.load('malaria', with_info=True,
                                  as_supervised=True, 
                                  shuffle_files=True, 
                                  split=['train'])

IM_SIZE = 224
@tf.function
def resize_rescale(image, label):
    return tf.image.resize(image, (IM_SIZE, IM_SIZE)) / 255.0, label

def splits(dataset, TRAIN_RATIO, VAL_RATIO):
    DATASET_SIZE = len(dataset)

    train_dataset = dataset.take(int(TRAIN_RATIO * DATASET_SIZE))
    val_test_dataset = dataset.skip(int(TRAIN_RATIO * DATASET_SIZE))
    val_dataset = val_test_dataset.take(int(VAL_RATIO * DATASET_SIZE))
    test_dataset = val_test_dataset.skip(int(VAL_RATIO * DATASET_SIZE))

    return train_dataset, val_dataset, test_dataset

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2

train_dataset, val_dataset, test_dataset = splits(dataset[0], TRAIN_RATIO, VAL_RATIO)

train_dataset = train_dataset.map(resize_rescale)
val_dataset = val_dataset.map(resize_rescale)
test_dataset = test_dataset.map(resize_rescale)

train_dataset = train_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)

callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='modelSaves/model.ckpt', 
                                                 save_weights_only=False,
                                                 save_best_only=True, 
                                                 verbose=1)]

filterList = [8, 16]
kernelSize = [3, 3]
strides = [1, 1]
padding = ['valid', 'valid']
activationCNN = ['relu', 'relu']
poolSize = [2, 2]
denseList = [100, 10, 1]
activationDense = ['relu', 'relu', 'sigmoid']

model = CustomCNNModel(filterList, kernelSize, strides, padding, activationCNN, poolSize, denseList, activationDense)
model(tf.zeros([1,IM_SIZE,IM_SIZE,3]))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
              loss=tf.keras.losses.BinaryCrossentropy(), 
              metrics=['accuracy'])

model.fit(train_dataset, 
          validation_data=val_dataset, 
          epochs=20, 
          callbacks=callbacks)

test_dataset = test_dataset.batch(32)
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")
