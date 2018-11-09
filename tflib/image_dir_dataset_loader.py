import numpy as np
import scipy.misc
import time
import os


def make_generator(path, batch_size, shuffle):
    epoch_count = [1]
    def get_epoch():
        n_files = len([name for name in os.listdir(path) if name.endswith('.png') or name.endswith('.jpg') or name.endswith('.jpeg')])
        print('Loader for ', path, ': ', n_files, 'files')
        path_suffix = path.split('/')[-1]
        latent_params_file = os.path.join(path, '../{}_latent_vectors.npy'.format(path_suffix))
        if os.path.isfile(latent_params_file):
            all_latent_params = np.load(latent_params_file)
        else:
            all_latent_params = np.zeros([n_files, 0])
        images = np.zeros((batch_size, 3, 64, 64), dtype='int32')
        latent_params = np.zeros((batch_size, all_latent_params.shape[1]), dtype=np.float32)
        files = range(n_files)
        if shuffle:
            random_state = np.random.RandomState(epoch_count[0])
            random_state.shuffle(files)
        epoch_count[0] += 1
        for n, i in enumerate(files):
            image = scipy.misc.imread("{}/{}.png".format(path, i))
            latent_params[n % batch_size] = all_latent_params[i, :]
            images[n % batch_size] = image.transpose(2,0,1)
            if n > 0 and n % batch_size == 0:
                yield (images, latent_params, )
    return get_epoch

def load(batch_size, data_dir='/home/ishaan/data/imagenet64', shuffle=True):
    return (
        make_generator(data_dir+'/train', batch_size, shuffle),
        make_generator(data_dir+'/test', batch_size, shuffle)
    )

def load_single_set(batch_size, data_dir='/home/ishaan/data/imagenet64', shuffle=True):
    return (
        make_generator(data_dir, batch_size, shuffle),
        None
    )

if __name__ == '__main__':
    train_gen, valid_gen = load(64)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print "{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0])
        if i == 1000:
            break
        t0 = time.time()
