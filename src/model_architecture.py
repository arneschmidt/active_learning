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
from utils.layers import Mil_Attention
import globals


def create_model(num_training_points: int):
    feature_extractor = create_feature_extactor()
    head = create_head(num_training_points, kl_factor=globals.config["model"]["head"]["bnn"]["kl_loss_factor"],
                       num_classes=globals.config['data']['num_classes'])
    
    model = Sequential([feature_extractor, head])
    return model


def create_wsi_level_model(num_training_points: int):
    #    ValueError: Shapes (None, 1) and (1, 6) are incompatible
    if globals.config['model']['wsi_level_model']['use']:
        attention_module = create_attention_module()
        head = create_head(num_training_points, kl_factor=0.1, num_classes=6)
        wsi_model = Sequential([attention_module, head])
    else:
        wsi_model = None
    return wsi_model


def create_feature_extactor():
    feature_extractor_type = globals.config["model"]["feature_extractor"]["type"]
    multiscale = globals.config["model"]["feature_extractor"]["multiscale"]
    image_size = globals.config["data"]["image_target_size"]
    weights = "imagenet"

    input = tf.keras.layers.Input(shape=(image_size[0], image_size[1], 3))

    if not multiscale:
        input_shape = (image_size[0], image_size[1], 3)
        x_complete = input
        x_center = None
    else:
        input_shape = (int(image_size[0] / 2), int(image_size[1] / 2), 3)
        x_complete = tf.keras.layers.experimental.preprocessing.Resizing(input_shape[0], input_shape[1])(input)
        x_center = tf.keras.layers.Cropping2D(cropping=int(image_size[0] / 4))(input)

    if feature_extractor_type == "mobilenetv2":
        x = MobileNetV2(include_top=False, input_shape=input_shape, weights=None, pooling='avg')(x_complete)
        if multiscale:
            center_model = MobileNetV2(include_top=False, input_shape=input_shape, weights=None, pooling='avg')
            center_model._name = 'center_cnn'
            x_center = center_model(x_center)
            x = tf.concat([x_center, x], axis=1)
    elif feature_extractor_type == "efficientnetb0":
        x = EfficientNetB0(include_top=False, input_shape=input_shape, weights=weights, pooling='avg')(x_complete)
        if multiscale:
            center_model = EfficientNetB0(include_top=False, input_shape=input_shape, weights=None, pooling='avg')
            center_model._name = 'center_cnn'
            x_center = center_model(x_center)
            x = tf.concat([x_center, x], axis=1)
    elif feature_extractor_type == "efficientnetb1":
        x = EfficientNetB1(include_top=False, input_shape=input_shape, weights=weights, pooling='avg')(x_complete)
        if multiscale:
            center_model = EfficientNetB1(include_top=False, input_shape=input_shape, weights=None, pooling='avg')
            center_model._name = 'center_cnn'
            x_center = center_model(x_center)
            x = tf.concat([x_center, x], axis=1)
    elif feature_extractor_type == "efficientnetb2":
        x = EfficientNetB2(include_top=False, input_shape=input_shape, weights=weights, pooling='avg')(x_complete)
        if multiscale:
            center_model = EfficientNetB2(include_top=False, input_shape=input_shape, weights=None, pooling='avg')
            center_model._name = 'center_cnn'
            x_center = center_model(x_center)
            x = tf.concat([x_center, x], axis=1)
    elif feature_extractor_type == "efficientnetb3":
        x = EfficientNetB3(include_top=False, input_shape=input_shape, weights=weights, pooling='avg')(x_complete)
        if multiscale:
            center_model = EfficientNetB3(include_top=False, input_shape=input_shape, weights=None, pooling='avg')
            center_model._name = 'center_cnn'
            x_center = center_model(x_center)
            x = tf.concat([x_center, x], axis=1)
    elif feature_extractor_type == "efficientnetb4":
        x = EfficientNetB4(include_top=False, input_shape=input_shape, weights=weights, pooling='avg')(x_complete)
        if multiscale:
            center_model = EfficientNetB4(include_top=False, input_shape=input_shape, weights=None, pooling='avg')
            center_model._name = 'center_cnn'
            x_center = center_model(x_center)
            x = tf.concat([x_center, x], axis=1)
    elif feature_extractor_type == "efficientnetb5":
        x = EfficientNetB5(include_top=False, input_shape=input_shape, weights=weights, pooling='avg')(x_complete)
        if multiscale:
            center_model = EfficientNetB5(include_top=False, input_shape=input_shape, weights=None, pooling='avg')
            center_model._name = 'center_cnn'
            x_center = center_model(x_center)
            x = tf.concat([x_center, x], axis=1)
    elif feature_extractor_type == "efficientnetb6":
        x = EfficientNetB6(include_top=False, input_shape=input_shape, weights=weights, pooling='avg')(x_complete)
        if multiscale:
            center_model = EfficientNetB6(include_top=False, input_shape=input_shape, weights=None, pooling='avg')
            center_model._name = 'center_cnn'
            x_center = center_model(x_center)
            x = tf.concat([x_center, x], axis=1)
    elif feature_extractor_type == "efficientnetb7":
        x = EfficientNetB7(include_top=False, input_shape=input_shape, weights=weights, pooling='avg')(x_complete)
        if multiscale:
            center_model = EfficientNetB7(include_top=False, input_shape=input_shape, weights=None, pooling='avg')
            center_model._name = 'center_cnn'
            x_center = center_model(x_center)
            x = tf.concat([x_center, x], axis=1)
    elif feature_extractor_type == "resnet50":
        x = ResNet50(include_top=False, input_shape=input_shape, weights=weights, pooling='avg')(x_complete)
        if multiscale:
            center_model = ResNet50(include_top=False, input_shape=input_shape, weights=None, pooling='avg')
            center_model._name = 'center_cnn'
            x_center = center_model(x_center)
            x = tf.concat([x_center, x], axis=1)
    else:
        raise Exception("Choose valid model architecture!")

    dropout_rate = globals.config["model"]["feature_extractor"]["dropout"]
    if dropout_rate > 0.0:
        x = Dropout(rate=dropout_rate)(x)

    if globals.config["model"]["feature_extractor"]["global_max_pooling"]:
        x = GlobalMaxPool2D()(x)
    if globals.config["model"]["feature_extractor"]["num_output_features"] > 0:
        activation = globals.config["model"]["feature_extractor"]["output_activation"]
        x = Dense(globals.config["model"]["feature_extractor"]["num_output_features"], activation=activation)(x)
    # feature_extractor.build(input_shape=input_shape)
    output = x
    feature_extractor = tf.keras.Model(inputs=input, outputs=output, name="feature_extractor")
    return feature_extractor


def create_head(num_training_points: int, kl_factor=1.0, num_classes=5):
        config = globals.config
        head_type = config["model"]["head"]["type"]

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

            kl_factor = tf.Variable(initial_value=kl_factor, trainable=False, dtype=tf.float32)
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

def create_attention_module():
    def attention_multiplication(i):
        # a = tf.ones_like(i[0])
        a = i[0]
        f = i[1]
        # tf.print('attention', a)
        # tf.print('features', f)
        a = tf.reshape(a, shape=[-1])
        out = tf.linalg.matvec(f, a, transpose_a=True)
        out = tf.reshape(out, [1, f.shape[1]])
        return out

    data_dims = globals.config["model"]["feature_extractor"]["num_output_features"]
    input = tf.keras.layers.Input(shape=(data_dims))
    a = Mil_Attention(L_dim=128, output_dim=0, name='instance_softmax', use_gated=True)(input)
    output = tf.keras.layers.Lambda(attention_multiplication)([a, input])
    attention_module = tf.keras.Model(inputs=input, outputs=output, name="attention_module")
    return attention_module

