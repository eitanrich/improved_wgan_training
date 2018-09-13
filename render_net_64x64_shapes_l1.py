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
# import tflib.cropped_celeba
import tflib.image_dir_dataset_loader
import tflib.ops.layernorm
import tflib.plot

DATA_DIR = '/cs/usr/eitanrich/phd-work-local/Datasets/NormalShapes/64-1-2018-09-06-10-21'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_64x64.py!')

MODE = 'l1' # dcgan, wgan, wgan-gp, lsgan
LATENT_DIM = 10
DIM = 64 # Model dimensionality
N_GPUS = 1 # Number of GPUs
BATCH_SIZE = 64 # Batch size. Must be a multiple of N_GPUS
ITERS = 20000 # How many iterations to train for
OUTPUT_DIM = 64*64*3 # Number of pixels in each iamge

lib.print_model_settings(locals().copy())

DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]

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

def GoodGenerator(n_samples, latent_params, dim=DIM, nonlinearity=tf.nn.relu):
    if latent_params is None:
        latent_params = tf.random_normal([n_samples, LATENT_DIM])

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
    all_real_data_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3, 64, 64])
    all_real_data_latent_params = tf.placeholder(tf.float32, shape=[BATCH_SIZE, LATENT_DIM])
    split_real_data_conv = tf.split(all_real_data_conv, len(DEVICES))
    split_real_data_latent_params = tf.split(all_real_data_latent_params, len(DEVICES))
    render_costs = []

    # for device_index, (device, real_data_conv) in enumerate(zip(DEVICES, split_real_data_conv)):
    for device_index in range(len(DEVICES)):
        device = DEVICES[device_index]
        real_data_conv = split_real_data_conv[device_index]
        real_latent_params = split_real_data_latent_params[device_index]
        with tf.device(device):
            real_data = tf.reshape(2*((tf.cast(real_data_conv, tf.float32)/255.)-.5), [BATCH_SIZE/len(DEVICES), OUTPUT_DIM])
            rendered_data = RenderNet(BATCH_SIZE/len(DEVICES), latent_params=real_latent_params)
            render_cost = tf.losses.absolute_difference(real_data, rendered_data)
            render_costs.append(render_cost)

    render_cost = tf.add_n(render_costs) / len(DEVICES)

    rendernet_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0., beta2=0.9).minimize(render_cost,
                                      var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)


    print('>>>>>>>>>>> For generating samples...')
    # For generating samples
    # fixed_latent_params = tf.constant(np.random.normal(size=(BATCH_SIZE, LATENT_DIM)).astype('float32'))
    # all_fixed_noise_samples = []
    # for device_index, device in enumerate(DEVICES):
    #     n_samples = BATCH_SIZE / len(DEVICES)
    #     all_fixed_noise_samples.append(RenderNet(n_samples, latent_params=fixed_latent_params[device_index*n_samples:(device_index+1)*n_samples]))
    # all_fixed_noise_samples = tf.concat(all_fixed_noise_samples, axis=0)
    def generate_image(iteration):
        # samples = session.run(all_fixed_noise_samples)
        samples = session.run(rendered_data, feed_dict={all_real_data_conv: _images, all_real_data_latent_params: _latent_params})
        samples = ((samples+1.)*(255.99/2)).astype('int32')
        print('Generated images ', samples.shape, ' - saving...')
        lib.save_images.save_images(samples.reshape((BATCH_SIZE, 3, 64, 64)), 'RenderNet_Shapes_samples_{}.png'.format(iteration))


    # Dataset iterator
    # train_gen, dev_gen = lib.small_imagenet.load(BATCH_SIZE, data_dir=DATA_DIR)
    train_gen, dev_gen = lib.image_dir_dataset_loader.load(BATCH_SIZE, data_dir=DATA_DIR)

    def inf_train_gen():
        while True:
            for (images, latent_params) in train_gen():
                yield images, latent_params

    saver = tf.train.Saver()
    snapshot_folder = './RenderNet_Shapes_model'
    if not os.path.isdir(snapshot_folder):
        os.makedirs(snapshot_folder)

    # Train loop
    session.run(tf.initialize_all_variables())

    # print '>>>>>>>>>>> Restoring session ...'
    # saver.restore(session, snapshot_folder + '/model.ckpt')

    print('>>>>>>>>>>> Train loop ...')
    gen = inf_train_gen()
    for iteration in xrange(ITERS):

        start_time = time.time()

        # Train generator
        _images, _latent_params = gen.next()
        _ = session.run(rendernet_train_op, feed_dict={all_real_data_conv: _images, all_real_data_latent_params: _latent_params})

        if iteration % 200 == 0:
            # if iteration == 0:
            #     tf.assign(fixed_latent_params, _latent_params)
            print(iteration)
            generate_image(iteration)
            saver.save(session, snapshot_folder + '/model.ckpt')
