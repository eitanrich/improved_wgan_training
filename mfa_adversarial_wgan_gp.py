import os, sys
sys.path.append(os.getcwd())

import time
import functools

import numpy as np
import tensorflow as tf
import sklearn.datasets

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
# import tflib.small_imagenet
import tflib.cropped_celeba
# import tflib.image_dir_dataset_loader
import tflib.ops.layernorm
import tflib.plot
from scipy.misc import imsave
from matplotlib import pyplot as plt
# import cv2
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from utils import mfa
from utils import mfa_utils

# Download 64x64 ImageNet at http://image-net.org/small/download.php and
# fill in the path to the extracted files here!
DATA_DIR = '/cs/usr/eitanrich/phd-work-local/Datasets/CelebA'
# DATA_DIR = '/cs/usr/eitanrich/phd-work-local/Datasets/NormalShapes/64-1-2018-09-06-10-21'
# DATA_DIR = '/cs/usr/eitanrich/phd-work-local/Datasets/NormalShapes/64-1-fixed-4-2018-09-16-15-48'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_64x64.py!')

VISUALIZE = False
GENERATE = True
INTERPOLATE = False
SAMPLES_TO_GENERATE = 20000
MODE = 'wgan-gp' # dcgan, wgan, wgan-gp, lsgan
LATENT_DIM = 10
DIM = 64 # Model dimensionality
CRITIC_ITERS = 5 # How many iterations to train the critic for
N_GPUS = 2 # Number of GPUs
BATCH_SIZE = 64 # Batch size. Must be a multiple of N_GPUS
ITERS = 50000 # How many iterations to train for
LAMBDA = 10 # Gradient penalty lambda hyperparameter
OUTPUT_DIM = 64*64*3 # Number of pixels in each iamge
OUTPUT_DIR = '../results-mfa-500c-20l'+MODE
# OUTPUT_DIR = './results-shapes-64-1-fixed-4-2018-09-16-15-48-'+MODE

if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


lib.print_model_settings(locals().copy())

def GeneratorAndDiscriminator():
    """
    Choose which generator and discriminator architecture to use by
    uncommenting one of these lines.
    """

    # For actually generating decent samples, use this one
    return MFAGenerator, GoodDiscriminator

    # Baseline (G: DCGAN, D: DCGAN)
    # return DCGANGenerator, DCGANDiscriminator

    # No BN and constant number of filts in G
    # return WGANPaper_CrippledDCGANGenerator, DCGANDiscriminator

    # 512-dim 4-layer ReLU MLP G
    # return FCGenerator, DCGANDiscriminator

    # No normalization anywhere
    # return functools.partial(DCGANGenerator, bn=False), functools.partial(DCGANDiscriminator, bn=False)

    # Gated multiplicative nonlinearities everywhere
    # return MultiplicativeDCGANGenerator, MultiplicativeDCGANDiscriminator

    # tanh nonlinearities everywhere
    # return functools.partial(DCGANGenerator, bn=True, nonlinearity=tf.tanh), \
    #        functools.partial(DCGANDiscriminator, bn=True, nonlinearity=tf.tanh)

    # 101-layer ResNet G and D
    # return ResnetGenerator, ResnetDiscriminator

    raise Exception('You must choose an architecture!')

DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return LeakyReLU(output)

def Normalize(name, axes, inputs):
    if ('Discriminator' in name) and (MODE == 'wgan-gp'):
        if axes != [0,2,3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
    else:
        return lib.ops.batchnorm.Batchnorm(name,axes,inputs,fused=True)

def pixcnn_gated_nonlinearity(a, b):
    return tf.sigmoid(a) * tf.tanh(b)

def SubpixelConv2D(*args, **kwargs):
    kwargs['output_dim'] = 4*kwargs['output_dim']
    output = lib.ops.conv2d.Conv2D(*args, **kwargs)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    return output

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def BottleneckResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = functools.partial(lib.ops.conv2d.Conv2D, stride=2)
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim/2)
        conv_1b       = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim/2, output_dim=output_dim/2, stride=2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim/2, output_dim=output_dim)
    elif resample=='up':
        conv_shortcut = SubpixelConv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim/2)
        conv_1b       = functools.partial(lib.ops.deconv2d.Deconv2D, input_dim=input_dim/2, output_dim=output_dim/2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim/2, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim/2)
        conv_1b       = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim/2,  output_dim=output_dim/2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim/2, output_dim=output_dim)

    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=1, inputs=output, he_init=he_init)
    output = tf.nn.relu(output)
    output = conv_1b(name+'.Conv1B', filter_size=filter_size, inputs=output, he_init=he_init)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=1, inputs=output, he_init=he_init, biases=False)
    output = Normalize(name+'.BN', [0,2,3], output)

    return shortcut + (0.3*output)

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = MeanPoolConv
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
    elif resample=='up':
        conv_shortcut = UpsampleConv
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = Normalize(name+'.BN1', [0,2,3], output)
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output, he_init=he_init, biases=False)
    output = Normalize(name+'.BN2', [0,2,3], output)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output, he_init=he_init)

    return shortcut + output


# ! Generators

def MFAGenerator(N, K, d, l, PI, A, MU):
    """
    Mixture of Factor Analyzers Generator - Eitan Richardson Sep 2018
    :param N: Number of samples (batch size)
    :param K: Number of components
    :param d: Data dimension
    :param l: Latent dimension
    :param PI: Mixing coefficients [K]
    :param A: Scale matrices (factor loadings) [K, d, l]
    :param MU: Component means [K, d]
    # :param D: Noise diagonal variance [K, d]
    :return: Samples drawn from the MFA [N, d]
    """
    Z_l = tf.random_normal([l, N])
#    Z_d = tf.random_normal([1, d, N])

    A_Z_l = tf.reshape(tf.matmul(tf.reshape(A, [d*K, l]), Z_l), [K, d, N])
    # D_Z_d = tf.reshape(D, [K, d, 1]) * Z_d

    # Y_all = A_Z_l + D_Z_d + tf.reshape(MU, [K, d, 1])
    Y_all = A_Z_l + tf.reshape(MU, [K, d, 1])

    mixture = tf.distributions.Multinomial(total_count=1.0, logits=PI)
    C = tf.transpose(mixture.sample(sample_shape=N))

    Y = tf.reduce_sum(Y_all * tf.reshape(C, [K, 1, N]), axis=0)

    # Convert from my MFA format to WGAN-GP format - channel order and value range
    Y_hat = tf.reshape(tf.transpose(tf.reshape(tf.transpose(Y), [N, 64, 64, 3]), [0, 3, 1, 2]), [N, d])*2.0 - 1.0
    return Y_hat


# def GoodGenerator(n_samples, noise=None, dim=DIM, nonlinearity=tf.nn.relu):
#     if noise is None:
#         noise = tf.random_normal([n_samples, LATENT_DIM])
#
#     output = lib.ops.linear.Linear('Generator.Input', LATENT_DIM, 4*4*8*dim, noise)
#     output = tf.reshape(output, [-1, 8*dim, 4, 4])
#
#     output = ResidualBlock('Generator.Res1', 8*dim, 8*dim, 3, output, resample='up')
#     output = ResidualBlock('Generator.Res2', 8*dim, 4*dim, 3, output, resample='up')
#     output = ResidualBlock('Generator.Res3', 4*dim, 2*dim, 3, output, resample='up')
#     output = ResidualBlock('Generator.Res4', 2*dim, 1*dim, 3, output, resample='up')
#
#     output = Normalize('Generator.OutputN', [0,2,3], output)
#     output = tf.nn.relu(output)
#     output = lib.ops.conv2d.Conv2D('Generator.Output', 1*dim, 3, 3, output)
#     output = tf.tanh(output)
#
#     return tf.reshape(output, [-1, OUTPUT_DIM])
#
# def FCGenerator(n_samples, noise=None, FC_DIM=512):
#     if noise is None:
#         noise = tf.random_normal([n_samples, LATENT_DIM])
#
#     output = ReLULayer('Generator.1', LATENT_DIM, FC_DIM, noise)
#     output = ReLULayer('Generator.2', FC_DIM, FC_DIM, output)
#     output = ReLULayer('Generator.3', FC_DIM, FC_DIM, output)
#     output = ReLULayer('Generator.4', FC_DIM, FC_DIM, output)
#     output = lib.ops.linear.Linear('Generator.Out', FC_DIM, OUTPUT_DIM, output)
#
#     output = tf.tanh(output)
#
#     return output
#
# def DCGANGenerator(n_samples, noise=None, dim=DIM, bn=True, nonlinearity=tf.nn.relu):
#     lib.ops.conv2d.set_weights_stdev(0.02)
#     lib.ops.deconv2d.set_weights_stdev(0.02)
#     lib.ops.linear.set_weights_stdev(0.02)
#
#     if noise is None:
#         noise = tf.random_normal([n_samples, LATENT_DIM])
#
#     output = lib.ops.linear.Linear('Generator.Input', LATENT_DIM, 4*4*8*dim, noise)
#     output = tf.reshape(output, [-1, 8*dim, 4, 4])
#     if bn:
#         output = Normalize('Generator.BN1', [0,2,3], output)
#     output = nonlinearity(output)
#
#     output = lib.ops.deconv2d.Deconv2D('Generator.2', 8*dim, 4*dim, 5, output)
#     if bn:
#         output = Normalize('Generator.BN2', [0,2,3], output)
#     output = nonlinearity(output)
#
#     output = lib.ops.deconv2d.Deconv2D('Generator.3', 4*dim, 2*dim, 5, output)
#     if bn:
#         output = Normalize('Generator.BN3', [0,2,3], output)
#     output = nonlinearity(output)
#
#     output = lib.ops.deconv2d.Deconv2D('Generator.4', 2*dim, dim, 5, output)
#     if bn:
#         output = Normalize('Generator.BN4', [0,2,3], output)
#     output = nonlinearity(output)
#
#     output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
#     output = tf.tanh(output)
#
#     lib.ops.conv2d.unset_weights_stdev()
#     lib.ops.deconv2d.unset_weights_stdev()
#     lib.ops.linear.unset_weights_stdev()
#
#     return tf.reshape(output, [-1, OUTPUT_DIM])
#
# def WGANPaper_CrippledDCGANGenerator(n_samples, noise=None, dim=DIM):
#     if noise is None:
#         noise = tf.random_normal([n_samples, LATENT_DIM])
#
#     output = lib.ops.linear.Linear('Generator.Input', LATENT_DIM, 4*4*dim, noise)
#     output = tf.nn.relu(output)
#     output = tf.reshape(output, [-1, dim, 4, 4])
#
#     output = lib.ops.deconv2d.Deconv2D('Generator.2', dim, dim, 5, output)
#     output = tf.nn.relu(output)
#
#     output = lib.ops.deconv2d.Deconv2D('Generator.3', dim, dim, 5, output)
#     output = tf.nn.relu(output)
#
#     output = lib.ops.deconv2d.Deconv2D('Generator.4', dim, dim, 5, output)
#     output = tf.nn.relu(output)
#
#     output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
#     output = tf.tanh(output)
#
#     return tf.reshape(output, [-1, OUTPUT_DIM])
#
# def ResnetGenerator(n_samples, noise=None, dim=DIM):
#     if noise is None:
#         noise = tf.random_normal([n_samples, LATENT_DIM])
#
#     output = lib.ops.linear.Linear('Generator.Input', LATENT_DIM, 4*4*8*dim, noise)
#     output = tf.reshape(output, [-1, 8*dim, 4, 4])
#
#     for i in range(6):
#         output = BottleneckResidualBlock('Generator.4x4_{}'.format(i), 8*dim, 8*dim, 3, output, resample=None)
#     output = BottleneckResidualBlock('Generator.Up1', 8*dim, 4*dim, 3, output, resample='up')
#     for i in range(6):
#         output = BottleneckResidualBlock('Generator.8x8_{}'.format(i), 4*dim, 4*dim, 3, output, resample=None)
#     output = BottleneckResidualBlock('Generator.Up2', 4*dim, 2*dim, 3, output, resample='up')
#     for i in range(6):
#         output = BottleneckResidualBlock('Generator.16x16_{}'.format(i), 2*dim, 2*dim, 3, output, resample=None)
#     output = BottleneckResidualBlock('Generator.Up3', 2*dim, 1*dim, 3, output, resample='up')
#     for i in range(6):
#         output = BottleneckResidualBlock('Generator.32x32_{}'.format(i), 1*dim, 1*dim, 3, output, resample=None)
#     output = BottleneckResidualBlock('Generator.Up4', 1*dim, dim/2, 3, output, resample='up')
#     for i in range(5):
#         output = BottleneckResidualBlock('Generator.64x64_{}'.format(i), dim/2, dim/2, 3, output, resample=None)
#
#     output = lib.ops.conv2d.Conv2D('Generator.Out', dim/2, 3, 1, output, he_init=False)
#     output = tf.tanh(output / 5.)
#
#     return tf.reshape(output, [-1, OUTPUT_DIM])
#
#
# def MultiplicativeDCGANGenerator(n_samples, noise=None, dim=DIM, bn=True):
#     if noise is None:
#         noise = tf.random_normal([n_samples, LATENT_DIM])
#
#     output = lib.ops.linear.Linear('Generator.Input', LATENT_DIM, 4*4*8*dim*2, noise)
#     output = tf.reshape(output, [-1, 8*dim*2, 4, 4])
#     if bn:
#         output = Normalize('Generator.BN1', [0,2,3], output)
#     output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])
#
#     output = lib.ops.deconv2d.Deconv2D('Generator.2', 8*dim, 4*dim*2, 5, output)
#     if bn:
#         output = Normalize('Generator.BN2', [0,2,3], output)
#     output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])
#
#     output = lib.ops.deconv2d.Deconv2D('Generator.3', 4*dim, 2*dim*2, 5, output)
#     if bn:
#         output = Normalize('Generator.BN3', [0,2,3], output)
#     output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])
#
#     output = lib.ops.deconv2d.Deconv2D('Generator.4', 2*dim, dim*2, 5, output)
#     if bn:
#         output = Normalize('Generator.BN4', [0,2,3], output)
#     output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])
#
#     output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
#     output = tf.tanh(output)
#
#     return tf.reshape(output, [-1, OUTPUT_DIM])

# ! Discriminators

def GoodDiscriminator(inputs, dim=DIM):
    output = tf.reshape(inputs, [-1, 3, 64, 64])
    output = lib.ops.conv2d.Conv2D('Discriminator.Input', 3, dim, 3, output, he_init=False)

    output = ResidualBlock('Discriminator.Res1', dim, 2*dim, 3, output, resample='down')
    output = ResidualBlock('Discriminator.Res2', 2*dim, 4*dim, 3, output, resample='down')
    output = ResidualBlock('Discriminator.Res3', 4*dim, 8*dim, 3, output, resample='down')
    output = ResidualBlock('Discriminator.Res4', 8*dim, 8*dim, 3, output, resample='down')

    output = tf.reshape(output, [-1, 4*4*8*dim])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)

    return tf.reshape(output, [-1])

def MultiplicativeDCGANDiscriminator(inputs, dim=DIM, bn=True):
    output = tf.reshape(inputs, [-1, 3, 64, 64])

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, dim*2, 5, output, stride=2)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = lib.ops.conv2d.Conv2D('Discriminator.2', dim, 2*dim*2, 5, output, stride=2)
    if bn:
        output = Normalize('Discriminator.BN2', [0,2,3], output)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*dim, 4*dim*2, 5, output, stride=2)
    if bn:
        output = Normalize('Discriminator.BN3', [0,2,3], output)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*dim, 8*dim*2, 5, output, stride=2)
    if bn:
        output = Normalize('Discriminator.BN4', [0,2,3], output)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = tf.reshape(output, [-1, 4*4*8*dim])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)

    return tf.reshape(output, [-1])


def ResnetDiscriminator(inputs, dim=DIM):
    output = tf.reshape(inputs, [-1, 3, 64, 64])
    output = lib.ops.conv2d.Conv2D('Discriminator.In', 3, dim/2, 1, output, he_init=False)

    for i in range(5):
        output = BottleneckResidualBlock('Discriminator.64x64_{}'.format(i), dim/2, dim/2, 3, output, resample=None)
    output = BottleneckResidualBlock('Discriminator.Down1', dim/2, dim*1, 3, output, resample='down')
    for i in range(6):
        output = BottleneckResidualBlock('Discriminator.32x32_{}'.format(i), dim*1, dim*1, 3, output, resample=None)
    output = BottleneckResidualBlock('Discriminator.Down2', dim*1, dim*2, 3, output, resample='down')
    for i in range(6):
        output = BottleneckResidualBlock('Discriminator.16x16_{}'.format(i), dim*2, dim*2, 3, output, resample=None)
    output = BottleneckResidualBlock('Discriminator.Down3', dim*2, dim*4, 3, output, resample='down')
    for i in range(6):
        output = BottleneckResidualBlock('Discriminator.8x8_{}'.format(i), dim*4, dim*4, 3, output, resample=None)
    output = BottleneckResidualBlock('Discriminator.Down4', dim*4, dim*8, 3, output, resample='down')
    for i in range(6):
        output = BottleneckResidualBlock('Discriminator.4x4_{}'.format(i), dim*8, dim*8, 3, output, resample=None)

    output = tf.reshape(output, [-1, 4*4*8*dim])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)

    return tf.reshape(output / 5., [-1])


def FCDiscriminator(inputs, FC_DIM=512, n_layers=3):
    output = LeakyReLULayer('Discriminator.Input', OUTPUT_DIM, FC_DIM, inputs)
    for i in range(n_layers):
        output = LeakyReLULayer('Discriminator.{}'.format(i), FC_DIM, FC_DIM, output)
    output = lib.ops.linear.Linear('Discriminator.Out', FC_DIM, 1, output)

    return tf.reshape(output, [-1])

def DCGANDiscriminator(inputs, dim=DIM, bn=True, nonlinearity=LeakyReLU):
    output = tf.reshape(inputs, [-1, 3, 64, 64])

    lib.ops.conv2d.set_weights_stdev(0.02)
    lib.ops.deconv2d.set_weights_stdev(0.02)
    lib.ops.linear.set_weights_stdev(0.02)

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, dim, 5, output, stride=2)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', dim, 2*dim, 5, output, stride=2)
    if bn:
        output = Normalize('Discriminator.BN2', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*dim, 4*dim, 5, output, stride=2)
    if bn:
        output = Normalize('Discriminator.BN3', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*dim, 8*dim, 5, output, stride=2)
    if bn:
        output = Normalize('Discriminator.BN4', [0,2,3], output)
    output = nonlinearity(output)

    output = tf.reshape(output, [-1, 4*4*8*dim])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)

    lib.ops.conv2d.unset_weights_stdev()
    lib.ops.deconv2d.unset_weights_stdev()
    lib.ops.linear.unset_weights_stdev()

    return tf.reshape(output, [-1])

Generator, Discriminator = GeneratorAndDiscriminator()

est_gmm = mfa.MFA()
est_gmm.load('init_gmm_celeba_full_kmfa_500c_20l')
G_PI, G_MU, G_A, G_D = mfa_utils.init_raw_parms_from_gmm_diag(est_gmm)
# G_PI, G_MU, G_A, G_D = mfa_utils.init_super_mfa_raw_prams_from_mfa(est_gmm)
# theta_G = [G_PI, G_MU, G_A, G_D]
theta_G = [G_PI, G_MU, G_A]
K = 500
d = 64*64*3
l = 20

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    all_real_data_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3, 64, 64])
    if tf.__version__.startswith('1.'):
        split_real_data_conv = tf.split(all_real_data_conv, len(DEVICES))
    else:
        split_real_data_conv = tf.split(0, len(DEVICES), all_real_data_conv)
    gen_costs, disc_costs = [],[]

    print('>>>>>>>>>>> Define Generator and Discriminator...')
    for device_index, (device, real_data_conv) in enumerate(zip(DEVICES, split_real_data_conv)):
        with tf.device(device):

            real_data = tf.reshape(2*((tf.cast(real_data_conv, tf.float32)/255.)-.5), [BATCH_SIZE//len(DEVICES), OUTPUT_DIM])
            # fake_data = Generator(BATCH_SIZE//len(DEVICES), K, d, l, G_PI, G_A, G_MU, G_D)
            fake_data = Generator(BATCH_SIZE//len(DEVICES), K, d, l, G_PI, G_A, G_MU)

            disc_real = Discriminator(real_data)
            disc_fake = Discriminator(fake_data)

            if MODE == 'wgan':
                gen_cost = -tf.reduce_mean(disc_fake)
                disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

            elif MODE == 'wgan-gp':
                gen_cost = -tf.reduce_mean(disc_fake)
                disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

                alpha = tf.random_uniform(
                    shape=[BATCH_SIZE//len(DEVICES),1],
                    minval=0.,
                    maxval=1.
                )
                differences = fake_data - real_data
                interpolates = real_data + (alpha*differences)
                gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                gradient_penalty = tf.reduce_mean((slopes-1.)**2)
                disc_cost += LAMBDA*gradient_penalty

            elif MODE == 'dcgan':
                try: # tf pre-1.0 (bottom) vs 1.0 (top)
                    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                                      labels=tf.ones_like(disc_fake)))
                    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                                        labels=tf.zeros_like(disc_fake)))
                    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real,
                                                                                        labels=tf.ones_like(disc_real)))                    
                except Exception as e:
                    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.ones_like(disc_fake)))
                    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.zeros_like(disc_fake)))
                    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real, tf.ones_like(disc_real)))                    
                disc_cost /= 2.

            elif MODE == 'lsgan':
                gen_cost = tf.reduce_mean((disc_fake - 1)**2)
                disc_cost = (tf.reduce_mean((disc_real - 1)**2) + tf.reduce_mean((disc_fake - 0)**2))/2.

            else:
                raise Exception()

            gen_costs.append(gen_cost)
            disc_costs.append(disc_cost)

    gen_cost = tf.add_n(gen_costs) / len(DEVICES)
    disc_cost = tf.add_n(disc_costs) / len(DEVICES)

    print('>>>>>>>>>>> Define optimizers...')
    # if MODE == 'wgan':
    #     gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost,
    #                                          var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
    #     disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_cost,
    #                                          var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)
    #
    #     clip_ops = []
    #     for var in lib.params_with_name('Discriminator'):
    #         clip_bounds = [-.01, .01]
    #         clip_ops.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))
    #     clip_disc_weights = tf.group(*clip_ops)
    #
    # elif MODE == 'wgan-gp':
    #     gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0., beta2=0.9).minimize(gen_cost,
    #                                       var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
    #     disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0., beta2=0.9).minimize(disc_cost,
    #                                        var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)
    #
    # elif MODE == 'dcgan':
    #     gen_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(gen_cost,
    #                                       var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
    #     disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(disc_cost,
    #                                        var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)
    #
    # elif MODE == 'lsgan':
    #     gen_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(gen_cost,
    #                                          var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
    #     disc_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(disc_cost,
    #                                           var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)
    #
    # else:
    #     raise Exception()

    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0., beta2=0.9).minimize(gen_cost,
                                var_list=theta_G, colocate_gradients_with_ops=True)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0., beta2=0.9).minimize(disc_cost,
                                var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

    saver = tf.train.Saver()
    snapshot_folder = OUTPUT_DIR + '/model'
    if not os.path.isdir(snapshot_folder):
        os.makedirs(snapshot_folder)

    if VISUALIZE:
        print('>>>>>>>>>>> Visualizing Trained Model...')
        session.run(tf.initialize_all_variables())
        saver.restore(session, snapshot_folder + '/model.ckpt')
        all_params = session.run(theta_G)
        K, d, l = all_params[2].shape
        # Adding D
        all_params.append(np.ones([K, d])*1e-3)
        est_gmm = mfa_utils.raw_to_gmm_diag(*all_params)
        est_gmm.save(os.path.join(OUTPUT_DIR, 'mfa_model'))
        mfa_utils.visualize_trained_model(est_gmm, ITERS, out_folder=OUTPUT_DIR, image_shape=(64, 64, 3))
        plt.show()
        exit()

    if GENERATE:
        session.run(tf.initialize_all_variables())
        saver.restore(session, snapshot_folder + '/model.ckpt')
        # tf_samples = Generator(BATCH_SIZE, K, d, l, G_PI, G_A, G_MU, G_D)
        tf_samples = Generator(BATCH_SIZE, K, d, l, G_PI, G_A, G_MU)

        def generate_samples():
            samples = session.run(tf_samples)
            return (samples+1.)*(255.99/2)

        out_folder = OUTPUT_DIR + '/generated'
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)

        idx = 0
        for i in range((SAMPLES_TO_GENERATE+BATCH_SIZE-1)//BATCH_SIZE):
            print('Generating batch', i)
            samples = generate_samples()
            for j in range(samples.shape[0]):
                img = samples[j, ...].reshape([3, 64, 64]).transpose([1, 2, 0])
                imsave(os.path.join(out_folder, '{}.png'.format(idx)), img)
                idx += 1
        exit()
    #
    # if INTERPOLATE:
    #     session.run(tf.initialize_all_variables())
    #     saver.restore(session, snapshot_folder + '/model.ckpt')
    #
    #     tf_latent_params = tf.placeholder(tf.float32, [BATCH_SIZE, LATENT_DIM])
    #     tf_samples = Generator(BATCH_SIZE, tf_latent_params)
    #
    #     def generate_samples(latent_params):
    #         samples = session.run(tf_samples, feed_dict={tf_latent_params: latent_params})
    #         return ((samples+1.)*(255.99/2)).astype('int32')
    #
    #     out_folder = OUTPUT_DIR + '/interpolation'
    #     if not os.path.isdir(out_folder):
    #         os.makedirs(out_folder)
    #
    #     for i in range(LATENT_DIM):
    #         print('Interpolating latent parameter', i)
    #         latent_params = np.zeros([BATCH_SIZE, LATENT_DIM])
    #         latent_params[:, i] = np.arange(-5.0, 4.99, 10.0/BATCH_SIZE)
    #         samples = generate_samples(latent_params)
    #         lib.save_images.save_images(samples.reshape((BATCH_SIZE, 3, 64, 64)), out_folder + '/interpolate_param_{}.png'.format(i))
    #     exit()

    print('>>>>>>>>>>> For generating samples...')
    # For generating samples
    all_gen_samples = []
    for device_index, device in enumerate(DEVICES):
        n_samples = BATCH_SIZE // len(DEVICES)
        # all_gen_samples.append(Generator(BATCH_SIZE//len(DEVICES), K, d, l, G_PI, G_A, G_MU, G_D))
        all_gen_samples.append(Generator(BATCH_SIZE//len(DEVICES), K, d, l, G_PI, G_A, G_MU))
    all_gen_samples = tf.concat(all_gen_samples, axis=0)

    def generate_image(iteration):
        samples = session.run(all_gen_samples)
        # mosaic = mfa_utils.images_to_mosaic(mfa_utils.to_cv_images(samples, 64, 64))
        # cv2.imwrite(OUTPUT_DIR + '/samples_{}.png'.format(iteration), mosaic)
        # mfa_utils.plot_samples(samples)
        # plt.savefig(OUTPUT_DIR + '/samples_{}.png'.format(iteration))

        samples = ((samples+1.)*(255.99/2)).astype('int32')
        # print('Generated images ', samples.shape, ' - saving...')
        lib.save_images.save_images(samples.reshape((BATCH_SIZE, 3, 64, 64)), OUTPUT_DIR + '/samples_{}.png'.format(iteration))


    # Dataset iterator
    # train_gen, dev_gen = lib.small_imagenet.load(BATCH_SIZE, data_dir=DATA_DIR)
    train_gen, dev_gen = lib.cropped_celeba.load(BATCH_SIZE, data_dir=DATA_DIR)

    def inf_train_gen():
        while True:
            for (images, ) in train_gen():
                yield images

    # # Save a batch of ground-truth samples
    # _x = inf_train_gen().next()
    # _x_r = session.run(real_data, feed_dict={real_data_conv: _x[:BATCH_SIZE/N_GPUS]})
    # _x_r = ((_x_r+1.)*(255.99/2)).astype('int32')
    # lib.save_images.save_images(_x_r.reshape((BATCH_SIZE/N_GPUS, 3, 64, 64)), OUTPUT_DIR + '/samples_groundtruth.png')

    # Train loop
    session.run(tf.initialize_all_variables())

    # print '>>>>>>>>>>> Restoring session ...'
    # saver.restore(session, snapshot_folder + '/model.ckpt')

    print('>>>>>>>>>>> Train loop ...')
    gen = inf_train_gen()
    for iteration in range(ITERS):
        # print('Iteration', iteration)
        start_time = time.time()

        # Train generator
        if iteration > 0:
            _ = session.run(gen_train_op)

        # Train critic
        if (MODE == 'dcgan') or (MODE == 'lsgan'):
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
        for i in range(disc_iters):
            _data = next(gen)
            _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={all_real_data_conv: _data})
            # if MODE == 'wgan':
            #     _ = session.run([clip_disc_weights])

        # lib.plot.plot('train disc cost', _disc_cost)
        # lib.plot.plot('time', time.time() - start_time)

        if iteration % 200 == 0:
            print('Iteration', iteration, ': generating images and saving snapshot')
            generate_image(iteration)
            saver.save(session, snapshot_folder + '/model.ckpt')

            # A_X = session.run(G_D)
            # plt.imshow(A_X)
            # plt.savefig(OUTPUT_DIR+'/A_X_{}.png'.format(iteration))

        # if (iteration < 5) or (iteration % 200 == 199):
        #     lib.plot.flush()

        # lib.plot.tick()
