import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

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


model = tf.keras.models.load_model('./modelSaves/model.ckpt')

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