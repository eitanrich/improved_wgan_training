import os, sys
sys.path.append(os.getcwd())

import time
import functools
import numpy as np
import tensorflow as tf
# from tensorflow.contrib.framework import list_variables
import sklearn.datasets
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
# import tflib.small_imagenet
# import tflib.cropped_celeba
import tflib.image_dir_dataset_loader
import tflib.ops.layernorm
# import tflib.plot
from matplotlib import pyplot as plt

# DATA_DIR = '/cs/usr/eitanrich/phd-work-local/Datasets/NormalShapes/64-1-2018-09-06-10-21'
# DATA_DIR = '/cs/usr/eitanrich/phd-work-local/Datasets/NormalShapes/64-1-fixed-4-2018-09-16-15-48'
# DATA_DIR = 'results-shapes-wgan-gp/generated'
DATA_DIR = 'results-shapes-64-1-fixed-4-2018-09-16-15-48-wgan-gp/generated'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_64x64.py!')

MODE = 'wgan-gp' # dcgan, wgan, wgan-gp, lsgan
LATENT_DIM = 10
DIM = 64 # Model dimensionality
BATCH_SIZE = 128 # Batch size. Must be a multiple of N_GPUS
ITERS = 400 # How many iterations to train for
OUTPUT_DIM = 64*64*3 # Number of pixels in each iamge
OUTPUT_DIR = './results-shapes-rendernet'
MAX_SAMPLES = 10000

lib.print_model_settings(locals().copy())

DEVICES = ['/gpu:0']

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

def GoodGenerator(n_samples, latent_params, dim=DIM):
    output = lib.ops.linear.Linear('Generator.Input', LATENT_DIM, 4*4*8*dim, latent_params)
    output = tf.reshape(output, [-1, 8*dim, 4, 4])

    output = ResidualBlock('Generator.Res1', 8*dim, 8*dim, 3, output, resample='up')
    output = ResidualBlock('Generator.Res2', 8*dim, 4*dim, 3, output, resample='up')
    output = ResidualBlock('Generator.Res3', 4*dim, 2*dim, 3, output, resample='up')
    output = ResidualBlock('Generator.Res4', 2*dim, 1*dim, 3, output, resample='up')

    output = Normalize('Generator.OutputN', [0,2,3], output)
    output = tf.nn.relu(output)
    output = lib.ops.conv2d.Conv2D('Generator.Output', 1*dim, 3, 3, output)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

RenderNet = GoodGenerator

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    # real_data_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3, 64, 64])
    real_data_conv = tf.Variable(np.zeros([BATCH_SIZE, 3, 64, 64], dtype=np.int32), dtype=tf.int32)
    latent_params = tf.Variable(np.zeros([BATCH_SIZE, LATENT_DIM]), dtype=tf.float32)
    with tf.device(DEVICES[0]):
        real_data = tf.reshape(2*((tf.cast(real_data_conv, tf.float32)/255.)-.5), [BATCH_SIZE, OUTPUT_DIM])
        rendered_data = RenderNet(BATCH_SIZE, latent_params=latent_params)
        render_cost = tf.losses.absolute_difference(real_data, rendered_data)

    # rendernet_inverse_op = tf.train.AdamOptimizer(learning_rate=1e-1, beta1=0., beta2=0.9).minimize(render_cost, var_list=[latent_params])
    rendernet_optimizer = tf.train.AdamOptimizer(learning_rate=0.3, name='Optimizer')
    rendernet_inverse_op = rendernet_optimizer.minimize(render_cost, var_list=[latent_params])
    # rendernet_inverse_op = tf.train.RMSPropOptimizer(learning_rate=0.1).minimize(render_cost, var_list=[latent_params])
    reset_optimizer_op = tf.variables_initializer(rendernet_optimizer.variables())

    # Dataset iterator
    # train_gen, dev_gen = lib.image_dir_dataset_loader.load(BATCH_SIZE, data_dir=DATA_DIR)
    train_gen, dev_gen = lib.image_dir_dataset_loader.load_single_set(BATCH_SIZE, data_dir=DATA_DIR)
    def single_epoch_train_gen():
        for (images, latent_params) in train_gen():
            yield images, latent_params

    print('>>>>>>>>>>> Restoring session ...')
    snapshot_folder = OUTPUT_DIR + '/model'
    # var_desc = list_variables(snapshot_folder)
    # var_list = [v[0] for v in var_desc]
    var_list=lib.params_with_name('Generator')
    saver = tf.train.Saver(var_list=var_list)

    session.run(tf.initialize_all_variables())
    saver.restore(session, snapshot_folder + '/model.ckpt')

    print('>>>>>>>>>>> Main loop ...')
    gen = single_epoch_train_gen()

    start_time = time.time()
    all_found_latent_params = []
    batch_num = 0
    for _images, _latent_params in gen:
        print('Images {} - {}...'.format(batch_num * BATCH_SIZE, (batch_num+1) * BATCH_SIZE - 1))
        # _ = session.run(RenderNet, feed_dict={real_data_conv: _images})
        real_data_conv.load(_images, session)
        latent_params.load(np.zeros([BATCH_SIZE, LATENT_DIM], dtype=np.float32))
        session.run(reset_optimizer_op)

        # lib.save_images.save_images(_images.reshape((BATCH_SIZE, 3, 64, 64)), 'Inverse_RenderNet_Shapes_GT.png')
        all_mean_z_diff = []
        for iteration in xrange(ITERS):
            z_found, _ = session.run([latent_params, rendernet_inverse_op])
            if _latent_params.size > 0:
                mean_z_diff = np.mean(np.abs(_latent_params - z_found))
                all_mean_z_diff.append(mean_z_diff)

        all_found_latent_params.append(z_found)
        if all_mean_z_diff:
            print('Final mean z diff:', all_mean_z_diff[-1])
            plt.plot(all_mean_z_diff)
            plt.grid(True)
            plt.pause(0.1)
        batch_num += 1
        if batch_num*BATCH_SIZE > MAX_SAMPLES:
            break
    found_latent_params = np.vstack(tuple(all_found_latent_params))
print('>>>>>>>>>>> Saving results ...')
# np.save(OUTPUT_DIR + '/found_latent_params.npy', found_latent_params)
np.save(OUTPUT_DIR + '/wgan_gp_found_latent_params-for-64-1-fixed-4-2018-09-16-15-48.npy', found_latent_params)
print('>>>>>>>>>>> Done')
if all_mean_z_diff:
    plt.show()
