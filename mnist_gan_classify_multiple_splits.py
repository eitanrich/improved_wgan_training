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
BATCH_SIZE = 100 # Batch size
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

sigma_x = 1.0

def calc_log_prob(x_err, z):
    return -1 * (np.sum(np.square(z), axis=2) + np.sum(np.square(x_err), axis=2) / (2.0 * sigma_x * sigma_x))


print 'Trying to reconstruct with the different models:'

# def ncc_loss(x, y):
#     mean_x = tf.reduce_mean(x, axis=1, keep_dims=True)
#     mean_y = tf.reduce_mean(y, axis=1, keep_dims=True)
#     std_x = tf....

train_data, dev_data, test_data = lib.mnist.load_now()
all_images, all_labels = test_data

num_samples = 10000     # MNIST Test size...
reconstruction_errors = np.zeros([num_samples, 10, OUTPUT_DIM])
# log_prob_x = np.zeros([num_samples, 10])
optimal_latents = np.zeros([num_samples, 10, LATENT_DIM])
labels = np.ones([num_samples, 10], dtype=int) * (-1)

max_log_division = 5
num_retries = 3
accuracies = np.zeros([max_log_division, num_retries])
for p in range(max_log_division):
    for t in range(num_retries):
        for digit in range(10):
            print '************* Reconstructing with digit', digit
            # train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, BATCH_SIZE)

            # Train loop
            with tf.Session() as session:
                real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
                latent_params = tf.Variable(np.zeros([BATCH_SIZE, LATENT_DIM]), dtype=tf.float32)
                generated_data = Generator(BATCH_SIZE, noise=latent_params)
                # reconstruction_loss = tf.losses.absolute_difference(real_data, generated_data)

                def calc_log_likelihood_x_loss(x, x_hat, z):
                    return tf.reduce_sum(tf.square(z)) + tf.reduce_sum(tf.square(x-x_hat)) / (2.0 * sigma_x * sigma_x)

                log_likelihood_x_loss = calc_log_likelihood_x_loss(real_data, generated_data, latent_params)
                generator_inverse_optimizer = tf.train.AdamOptimizer(learning_rate=0.005, name='Optimizer')
                # generator_inverse_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005, name='Optimizer')
                generator_inverse_op = generator_inverse_optimizer.minimize(log_likelihood_x_loss, var_list=[latent_params])
                reset_optimizer_op = tf.variables_initializer(generator_inverse_optimizer.variables())

                gen_params = lib.params_with_name('Generator')
                saver = tf.train.Saver(var_list=gen_params)
                snapshot_folder = 'mnist_splits/model_{}d_{}p_{}t/model'.format(digit, p, t)

                session.run(tf.initialize_all_variables())
                saver.restore(session, tf.train.latest_checkpoint(snapshot_folder))

                for idx in range(0, num_samples-BATCH_SIZE+1, BATCH_SIZE):
                    end_idx = idx+BATCH_SIZE
                    print 'Processing images %d - %d' % (idx, end_idx-1)
                    # Run Z optimization iterations...
                    session.run(reset_optimizer_op)
                    for iteration in xrange(ITERS):
                        z_estimated, _loss, _ = session.run([latent_params, log_likelihood_x_loss, generator_inverse_op],
                                                                           feed_dict={real_data: all_images[idx:end_idx]})
                    reconstruction_errors[idx:end_idx, digit] = all_images[idx:end_idx] - session.run(generated_data)
                    optimal_latents[idx:end_idx, digit] = session.run(latent_params)
                    labels[idx:end_idx, digit] = all_labels[idx:end_idx]
            # Classify...
        log_probs = calc_log_prob(reconstruction_errors, optimal_latents)
        predictions = np.argmax(log_probs, axis=1)
        accuracies[p, t] = (np.sum(predictions[:num_samples] == labels[:num_samples, 0]))/float(num_samples)
        np.save('mnist_splits/accuracies.npy', accuracies)
print 'Done.'

