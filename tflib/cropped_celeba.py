import numpy as np
import scipy.misc
import time
import os

def make_generator(path, image_list, batch_size):
    epoch_count = [1]

    def get_epoch():
        images = np.zeros((batch_size, 3, 64, 64), dtype='int32')
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(image_list)
        epoch_count[0] += 1
        for n, image_name in enumerate(image_list):
            image = scipy.misc.imread(os.path.join(path, image_name))
            image = image[50:50+128, 25:25+128, :]
            image = scipy.misc.imresize(image, (64, 64))
            images[n % batch_size] = image.transpose(2, 0, 1)
            if n > 0 and n % batch_size == 0:
                yield (images,)
    return get_epoch


def get_train_validation_lists(data_dir):
    train_image_list = []
    val_image_list = []
    with open(os.path.join(data_dir, 'list_eval_partition.txt')) as f:
        for line in f:
            if int(line.split(' ')[1]) == 0:
                train_image_list.append(line.split(' ')[0])
            else:
                val_image_list.append(line.split(' ')[0])
    return train_image_list, val_image_list


def load(batch_size, data_dir='/home/ishaan/data/imagenet64'):
    train_list, val_list = get_train_validation_lists(data_dir)
    image_dir = os.path.join(data_dir, 'img_align_celeba')
    return (
        make_generator(image_dir, train_list, batch_size),
        make_generator(image_dir, val_list, batch_size)
    )

if __name__ == '__main__':
    train_gen, valid_gen = load(64, data_dir='/cs/labs/yweiss/eitanrich/Datasets/CelebA/')
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print("{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0]))
        if i == 1000:
            break
        t0 = time.time()
