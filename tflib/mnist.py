import numpy

import os
import urllib
import gzip
import pickle

def mnist_generator(data, batch_size, digits_filter, div_by=None):
    images, targets = data

    if digits_filter is not None:
        relevant_samples = numpy.isin(targets, digits_filter)
        images = images[relevant_samples]
        targets = targets[relevant_samples]
    print('Samples shape =', images.shape)
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(images)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(targets)
    if div_by is not None:
        limit = int(round(images.shape[0]/div_by))
        print("WARNING ONLY 1/{} = {} MNIST DIGITS".format(div_by, limit))
        images = images.astype('float32')[:limit]
        targets = targets.astype('int32')[:limit]

    def get_epoch():
        rng_state = numpy.random.get_state()
        numpy.random.shuffle(images)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(targets)

        num_samples = images.shape[0] - images.shape[0]%batch_size
        image_batches = images[:num_samples].reshape(-1, batch_size, 784)
        target_batches = targets[:num_samples].reshape(-1, batch_size)

        for i in xrange(len(image_batches)):
            yield (numpy.copy(image_batches[i]), numpy.copy(target_batches[i]))

    return get_epoch

def load(batch_size, test_batch_size, digits_filter=None, div_by=None):
    filepath = '/tmp/mnist.pkl.gz'
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

    if not os.path.isfile(filepath):
        print("Couldn't find MNIST dataset in /tmp, downloading...")
        urllib.urlretrieve(url, filepath)

    with gzip.open('/tmp/mnist.pkl.gz', 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f)

    return (
        mnist_generator(train_data, batch_size, digits_filter, div_by),
        mnist_generator(dev_data, test_batch_size, digits_filter, div_by),
        mnist_generator(test_data, test_batch_size, digits_filter, div_by)
    )

def load_now(digits_filter=None):
    filepath = '/tmp/mnist.pkl.gz'
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

    if not os.path.isfile(filepath):
        print("Couldn't find MNIST dataset in /tmp, downloading...")
        urllib.urlretrieve(url, filepath)

    with gzip.open('/tmp/mnist.pkl.gz', 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f)

    return train_data, dev_data, test_data

