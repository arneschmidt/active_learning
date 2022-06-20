import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from typing import Dict
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, \
    EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten, GlobalMaxPool2D, SeparableConv2D
from tensorflow_probability.python.layers import util as tfp_layers_util

import globals

def create_model(num_training_points: int):
    feature_extractor = create_feature_extactor()
    head = create_head(num_training_points)
    
    model = Sequential([feature_extractor, head])
    return model

def create_feature_extactor():
    input_shape = (globals.config["data"]["image_target_size"][0], globals.config["data"]["image_target_size"][1], 3)
    feature_extractor_type = globals.config["model"]["feature_extractor"]["type"]

    weights = "imagenet"
    feature_extractor = Sequential(name='feature_extractor')
    if feature_extractor_type == "mobilenetv2":
        feature_extractor.add(MobileNetV2(include_top=False, input_shape=input_shape, weights=None, pooling='avg'))
    elif feature_extractor_type == "efficientnetb0":
        feature_extractor.add(EfficientNetB0(include_top=False, input_shape=input_shape, weights=weights, pooling='avg'))
    elif feature_extractor_type == "efficientnetb1":
        feature_extractor.add(EfficientNetB1(include_top=False, input_shape=input_shape, weights=weights, pooling='avg'))
    elif feature_extractor_type == "efficientnetb2":
        feature_extractor.add(EfficientNetB2(include_top=False, input_shape=input_shape, weights=weights, pooling='avg'))
    elif feature_extractor_type == "efficientnetb3":
        feature_extractor.add(EfficientNetB3(include_top=False, input_shape=input_shape, weights=weights, pooling='avg'))
    elif feature_extractor_type == "efficientnetb4":
        feature_extractor.add(EfficientNetB4(include_top=False, input_shape=input_shape, weights=weights, pooling='avg'))
    elif feature_extractor_type == "efficientnetb5":
        feature_extractor.add(EfficientNetB5(include_top=False, input_shape=input_shape, weights=weights, pooling='avg'))
    elif feature_extractor_type == "efficientnetb6":
        feature_extractor.add(EfficientNetB6(include_top=False, input_shape=input_shape, weights=weights, pooling='avg'))
    elif feature_extractor_type == "efficientnetb7":
        feature_extractor.add(EfficientNetB7(include_top=False, input_shape=input_shape, weights=weights, pooling='avg'))
    elif feature_extractor_type == "resnet50":
        feature_extractor.add(ResNet50(include_top=False, input_shape=input_shape, weights=weights, pooling='avg'))
    elif feature_extractor_type == "simple_cnn":
        feature_extractor.add(tf.keras.layers.Input(shape=input_shape))
        feature_extractor.add(SeparableConv2D(64, kernel_size=3, activation='relu', input_shape=input_shape))
        for i in range(3):
            feature_extractor.add(SeparableConv2D(32, kernel_size=3, activation='relu'))
            feature_extractor.add(SeparableConv2D(32, kernel_size=3, activation='relu'))
            feature_extractor.add(MaxPool2D(pool_size=(2, 2)))
        feature_extractor.add(SeparableConv2D(32, kernel_size=3, activation='relu'))
        feature_extractor.add(SeparableConv2D(32, kernel_size=3, activation='relu'))
    elif feature_extractor_type == "fsconv":
        feature_extractor.add(tf.keras.layers.Input(shape=input_shape))
        feature_extractor.add(Conv2D(32, kernel_size=3, activation='relu'))
        feature_extractor.add(MaxPool2D(strides=(2, 2)))
        feature_extractor.add(Conv2D(124, kernel_size=3, activation='relu'))
        feature_extractor.add(MaxPool2D(strides=(2, 2)))
        feature_extractor.add(Conv2D(512, kernel_size=3, activation='relu'))
        feature_extractor.add(MaxPool2D(strides=(2, 2)))
    else:
        raise Exception("Choose valid model architecture!")

    dropout_rate = globals.config["model"]["feature_extractor"]["dropout"]
    if dropout_rate > 0.0:
        feature_extractor.add(Dropout(rate=dropout_rate))

    if globals.config["model"]["feature_extractor"]["global_max_pooling"]:
        feature_extractor.add(GlobalMaxPool2D())
    if globals.config["model"]["feature_extractor"]["num_output_features"] > 0:
        activation = globals.config["model"]["feature_extractor"]["output_activation"]
        feature_extractor.add(Dense(globals.config["model"]["feature_extractor"]["num_output_features"], activation=activation))
    # feature_extractor.build(input_shape=input_shape)
    return feature_extractor


def create_head(num_training_points: int):
        config = globals.config
        head_type = config["model"]["head"]["type"]

        num_classes = config['data']['num_classes']
        if head_type == "deterministic":
            hidden_units = config["model"]["head"]["deterministic"]["number_hidden_units"]
            dropout_rate = config["model"]["head"]["deterministic"]["dropout"]
            head = Sequential(name='head')
            head.add(Dropout(rate=dropout_rate))
            head.add(Dense(hidden_units, activation="relu"))
            if config["model"]["head"]["deterministic"]["extra_layer"]:
                head.add(Dense(hidden_units, activation="relu"))
            head.add(Dense(int(num_classes), activation="softmax"))

        elif head_type == "bnn":
            number_hidden_units = config["model"]["head"]["bnn"]["number_hidden_units"]
            activation = 'relu'
            weight_std = config["model"]["head"]["bnn"]["weight_std"]
            weight_std_softplusinv = tfp.math.softplus_inverse(weight_std)

            kl_factor =  tf.Variable(initial_value=config["model"]["head"]["bnn"]["kl_loss_factor"], trainable=False, dtype=tf.float32)
            tfd = tfp.distributions
            # scaling of KL divergence to batch is included already, scaling to dataset size needs to be done
            kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) * tf.cast(kl_factor / num_training_points, dtype=tf.float32))   #  tf.cast(num_training_points, dtype=tf.float32))

            kernel_posterior_fn = tfp_layers_util.default_mean_field_normal_fn(
                untransformed_scale_initializer=tf.keras.initializers.RandomNormal(mean=weight_std_softplusinv, stddev=0.1)) # softplus transformed init param

            tensor_fn = (lambda d: d.sample())

            layers = [tfp.layers.DenseReparameterization(activation=activation, units=number_hidden_units,
                                                         kernel_posterior_fn=kernel_posterior_fn,
                                                         kernel_divergence_fn=kl_divergence_function,
                                                         bias_divergence_fn=kl_divergence_function,
                                                         kernel_posterior_tensor_fn=tensor_fn,
                                                         bias_posterior_tensor_fn=tensor_fn
                                                         )]
            if config["model"]["head"]["bnn"]["extra_layer"]:
                layers.append(tfp.layers.DenseReparameterization(activation=activation, units=number_hidden_units,
                                                                 kernel_posterior_fn=kernel_posterior_fn,
                                                                 kernel_divergence_fn=kl_divergence_function,
                                                                 bias_divergence_fn=kl_divergence_function,
                                                                 kernel_posterior_tensor_fn=tensor_fn,
                                                                 bias_posterior_tensor_fn=tensor_fn
                                                                 ))
            layers.append(tfp.layers.DenseReparameterization(activation="softmax", units=int(num_classes),
                                                             kernel_posterior_fn=kernel_posterior_fn,
                                                             kernel_divergence_fn=kl_divergence_function,
                                                             bias_divergence_fn=kl_divergence_function,
                                                             kernel_posterior_tensor_fn=tensor_fn,
                                                             bias_posterior_tensor_fn=tensor_fn
                                                             ))

            head = tf.keras.Sequential(layers, name='head')
        else:
            raise Exception("Choose valid model head!")
        return head

