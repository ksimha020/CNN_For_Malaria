Project Overview

This project was developed to enhance my skills with TensorFlow and has resulted in a robust model.
Model Architecture

The model is designed with two custom classes:

    Convolutional Layers Class: This class handles the convolutional operations, consisting of two convolutional layers. Each layer is followed by batch normalization and max pooling to improve feature extraction and stability.
    Dropout and Dense Network Class: This class includes a dropout layer to prevent overfitting and a dense network that condenses the feature representation to a single node. The final node outputs a confidence interval indicating whether a sample is parasitized or unparasitized.

Performance

The model has achieved a maximum accuracy of 94.12% on raw, unseen data, reflecting its effectiveness in distinguishing between parasitized and unparasitized samples.
