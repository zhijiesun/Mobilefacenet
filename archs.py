from keras.regularizers import l2
from keras.models import Model
from keras.layers import *
from arcface import *
import tensorflow as tf


weight_decay=0.0005
def InvertedResidualBlock(x, expand, out_channels, repeats, stride, weight_decay, block_id):

    '''

    This function defines a sequence of 1 or more identical layers, referring to Table 2 in the original paper.

    :param x: Input Keras tensor in (B, H, W, C_in)

    :param expand: expansion factor in bottlenect residual block

    :param out_channels: number of channels in the output tensor

    :param repeats: number of times to repeat the inverted residual blocks including the one that changes the dimensions.

    :param stride: stride for the 1x1 convolution

    :param weight_decay: hyperparameter for the l2 penalty

    :param block_id: as its name tells

    :return: Output tensor (B, H_new, W_new, out_channels)



    '''

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    in_channels = K.int_shape(x)[channel_axis]

    x = Conv2D(expand * in_channels, 1, padding='same', strides=stride, use_bias=False,

                kernel_regularizer=l2(weight_decay), name='conv_%d_0' % block_id)(x)

    x = BatchNormalization()(x)
    x = PReLU()(x)

    x = DepthwiseConv2D((3, 3),

                        padding='same',

                        depth_multiplier=1,

                        strides=1,

                        use_bias=False,

                        kernel_regularizer=l2(weight_decay),

                        name='conv_dw_%d_0' % block_id )(x)

    x = BatchNormalization()(x)
    x = PReLU()(x)

    x = Conv2D(out_channels, 1, padding='same', strides=1, use_bias=False,

               kernel_regularizer=l2(weight_decay), name='conv_bottleneck_%d_0' % block_id)(x)

    x = BatchNormalization()(x)



    for i in range(1, repeats):

        x1 = Conv2D(expand*out_channels, 1, padding='same', strides=1, use_bias=False,

                    kernel_regularizer=l2(weight_decay), name='conv_%d_%d' % (block_id, i))(x)

        x = BatchNormalization()(x)
        x = PReLU()(x)

        x1 = DepthwiseConv2D((3, 3),

                            padding='same',

                            depth_multiplier=1,

                            strides=1,

                            use_bias=False,

                            kernel_regularizer=l2(weight_decay),

                            name='conv_dw_%d_%d' % (block_id, i))(x1)

        x = BatchNormalization()(x)
        x = PReLU()(x)

        x1 = Conv2D(out_channels, 1, padding='same', strides=1, use_bias=False,

                    kernel_regularizer=l2(weight_decay),name='conv_bottleneck_%d_%d' % (block_id, i))(x1)

        x1 = BatchNormalization(axis=channel_axis, epsilon=1e-5, momentum=0.9, name='conv_bottlenet_%d_%d_bn' % (block_id, i))(x1)

        x = Add()([x, x1])

    return x

def regular_conv(x, filters, ks, p, s, k_init, use_b=False):
  x = Conv2D(filters, ks, padding=p, strides=s,
     kernel_initializer=k_init, kernel_regularizer=regularizers.l2(weight_decay), use_bias=use_b)(x)
  x = BatchNormalization()(x)
  x = PReLU()(x)

  return x

def depth_conv(x, filters, ks, p, s, k_init, use_b=False):
  x = DepthwiseConv2D(ks, padding=p, strides=s,
     kernel_initializer=k_init, kernel_regularizer=regularizers.l2(weight_decay), use_bias=use_b)(x)
  x = BatchNormalization()(x)
  x = PReLU()(x)
  return x


def valid_conv(x, filters, ks, p, s, k_init, use_b=False):
  x = Conv2D(filters, ks, padding=p, strides=s,
     kernel_initializer=k_init, kernel_regularizer=regularizers.l2(weight_decay), use_bias=use_b)(x)
  x = BatchNormalization()(x)
  x = PReLU()(x)

  return x


def mobilefacenet():
    input = Input(shape=(112, 96, 3))
    # parameter order: filters_num, kernel_size, padding, stride, kernel_initializer, bias_term
    x = regular_conv(input, 64, (3, 3), 'same', (2, 2), 'glorot_uniform', False)
    x = depth_conv(x, 64, (3, 3), 'valid', (1, 1), 'glorot_uniform', False)
    x = InvertedResidualBlock(x, 5,  64, 2, (2,2),  weight_decay, 1)
    x = InvertedResidualBlock(x, 1, 128, 4, (2, 2), weight_decay, 2)
    x = InvertedResidualBlock(x, 6, 128, 2, (1, 1), weight_decay, 3)
    x = InvertedResidualBlock(x, 1, 128, 4, (2, 2), weight_decay, 4)
    x = InvertedResidualBlock(x, 2, 128, 2, (1, 1), weight_decay, 5)
    x = valid_conv(x, 512, (1, 1), 'valid', (1, 1), 'glorot_uniform', False)
    x = depth_conv(x, 512, (7, 6), 'valid', (1, 1), 'glorot_uniform', False)

    x = valid_conv(x, 128, (1, 1), 'valid', (1, 1), 'glorot_uniform', False)
    print(x.shape)
    x = Flatten()(x)
    output = Dense(128, use_bias=False, kernel_initializer='glorot_uniform')(x)

    return Model(input, output)

def mobilefacenet_arcface():
    input = Input(shape=(112, 96, 3))
    y = Input(shape=(12,))
    # parameter order: filters_num, kernel_size, padding, stride, kernel_initializer, bias_term
    x = regular_conv(input, 64, (3, 3), 'same', (2, 2), 'glorot_uniform', False)
    x = depth_conv(x, 64, (3, 3), 'valid', (1, 1), 'glorot_uniform', False)
    x = InvertedResidualBlock(x, 5, 64, 2, (2, 2), weight_decay, 1)
    x = InvertedResidualBlock(x, 1, 128, 4, (2, 2), weight_decay, 2)
    x = InvertedResidualBlock(x, 6, 128, 2, (1, 1), weight_decay, 3)
    x = InvertedResidualBlock(x, 1, 128, 4, (2, 2), weight_decay, 4)
    x = InvertedResidualBlock(x, 2, 128, 2, (1, 1), weight_decay, 5)
    x = valid_conv(x, 512, (1, 1), 'valid', (1, 1), 'glorot_uniform', False)
    x = depth_conv(x, 512, (7, 6), 'valid', (1, 1), 'glorot_uniform', False)

    x = valid_conv(x, 128, (1, 1), 'valid', (1, 1), 'glorot_uniform', False)
    x = Flatten()(x)
    x = Dense(128, use_bias=False, kernel_initializer='glorot_uniform')(x)
    output = ArcFace(12, regularizer=regularizers.l2(weight_decay))([x, y])
    return Model([input, y], output)


