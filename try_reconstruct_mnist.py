import os, sys
sys.path.append(os.getcwd())

import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.mnist
import tflib.plot

MODE = 'wgan-gp' # dcgan, wgan, or wgan-gp
DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 3000 # How many generator iterations to train for
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)
LATENT_DIM = 128

lib.print_model_settings(locals().copy())

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)

def Generator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, LATENT_DIM])

    output = lib.ops.linear.Linear('Generator.Input', LATENT_DIM, 4*4*4*DIM, noise)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = output[:,:,:7,:7]

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 1, 5, output)
    output = tf.nn.sigmoid(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

# Dataset iterator
train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, BATCH_SIZE)
def inf_train_gen():
    while True:
        for images,targets in test_gen():
            yield images

print 'Loading a batch of real data (train samples):'
gen = inf_train_gen()
_real_data = gen.next()

# out_folder = 'per_class_models/reconstructions'
# if not os.path.isdir(out_folder):
#     os.makedirs(out_folder)

print 'Trying to reconstruct with the different models:'

# def ncc_loss(x, y):
#     mean_x = tf.reduce_mean(x, axis=1, keep_dims=True)
#     mean_y = tf.reduce_mean(y, axis=1, keep_dims=True)
#     std_x = tf....

reconstructions = np.zeros([10, BATCH_SIZE, OUTPUT_DIM])
noisy_z_reconstructions = np.zeros([10, 8, BATCH_SIZE, OUTPUT_DIM])
optimal_latents = np.zeros([10, BATCH_SIZE, LATENT_DIM])
for digit in range(10):
    print '************* Reconstructing with digit', digit

    # Train loop
    with tf.Session() as session:
        real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
        latent_params = tf.Variable(np.zeros([BATCH_SIZE, LATENT_DIM]), dtype=tf.float32)
        generated_data = Generator(BATCH_SIZE, noise=latent_params)
        # reconstruction_loss = tf.losses.absolute_difference(real_data, generated_data)

        def calc_log_likelihood_x_loss(x, x_hat, z):
            sigma_x = 0.5
            return tf.reduce_sum(tf.square(z)) + tf.reduce_sum(tf.square(x-x_hat)) / (2.0 * sigma_x * sigma_x)

        log_likelihood_x_loss = calc_log_likelihood_x_loss(real_data, generated_data, latent_params)
        generator_inverse_optimizer = tf.train.AdamOptimizer(learning_rate=0.005, name='Optimizer')
        # generator_inverse_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005, name='Optimizer')
        generator_inverse_op = generator_inverse_optimizer.minimize(log_likelihood_x_loss, var_list=[latent_params])

        gen_params = lib.params_with_name('Generator')
        saver = tf.train.Saver(var_list=gen_params)
        snapshot_folder = './per_class_models_latent_dim_'+str(LATENT_DIM)+'/digit_' + str(digit) + '/model'

        session.run(tf.initialize_all_variables())
        saver.restore(session, tf.train.latest_checkpoint(snapshot_folder))

        for iteration in xrange(ITERS):
            z_estimated, _loss, _ = session.run([latent_params, log_likelihood_x_loss, generator_inverse_op],
                                                               feed_dict={real_data: _real_data})
            if iteration%50 == 0:
                print iteration, _loss

        reconstructions[digit] = session.run(generated_data)
        optimal_latents[digit] = session.run(latent_params)

        # Add reconstruction with noise in z
        for i in range(noisy_z_reconstructions.shape[1]):
            latents = optimal_latents[digit] + np.random.normal(loc=0.0, scale=0.2, size=optimal_latents[digit].shape)
            # latents = np.zeros(optimal_latents[digit].shape)
            noisy_z_reconstructions[digit, i] = session.run(generated_data, feed_dict={latent_params: latents})

print 'saving results...'
np.save('real_digits.npy', _real_data)
np.save('reconstructed_digits.npy', reconstructions)
np.save('estimated_posterior_latents.npy', optimal_latents)
np.save('reconstructed_digits_with_noisy_latents.npy', noisy_z_reconstructions)
