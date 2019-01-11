import numpy as np
import tflib as lib
import tflib.mnist
from matplotlib import pyplot as plt
from cv2 import imwrite

def calc_log_prob(x_err, z):
    sigma_x = 2.0
    return -1 * (np.sum(np.square(z), axis=2) + np.sum(np.square(x_err), axis=2) / (2.0 * sigma_x * sigma_x))

reconstruction_errors = np.load('reconstruction_errors_2.npy')
optimal_latents = np.load('estimated_posterior_latents_2.npy')
labels = np.load('labels_2.npy')

log_probs = calc_log_prob(reconstruction_errors, optimal_latents)
predictions = np.argmax(log_probs, axis=1)
assert np.all(np.max(labels, axis=1) == np.min(labels, axis=1))

num_samples = 9900
print predictions[:num_samples]
print labels[:num_samples, 0]
accuracy = (np.sum(predictions[:num_samples] == labels[:num_samples, 0]))/float(num_samples)
print accuracy

# Visualize the errors
train_data, dev_data, test_data = lib.mnist.load_now()
all_images, all_labels = test_data
assert np.all(labels[:num_samples, 0] == all_labels[:num_samples])

misclassified = np.nonzero(predictions[:num_samples] != labels[:num_samples, 0])[0]
n = min(len(misclassified), 50)
w = 28
k = 10
mosaic = np.zeros([w*(k+1)+1, w*(n+1)])
mosaic[w, :] = 1
for i in range(n):
    mosaic[:w, i*w:(i+1)*w] = all_images[misclassified[i]].reshape([w, w])
    for j in range(k):
        mosaic[(j+1)*w+1:(j+2)*w+1, i*w:(i+1)*w] = (all_images[misclassified[i]] -
                                                    reconstruction_errors[misclassified[i], j]).reshape([w, w])
    predicted = predictions[misclassified[i]]
    mosaic[(predicted+1)*w+1:(predicted+2)*w+1, i*w] = 1

imwrite('classification_errors.png', mosaic*255)
