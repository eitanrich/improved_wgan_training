import numpy as np

def calc_log_prob(x_err, z):
    sigma_x = 1.0
    return -1 * (np.sum(np.square(z), axis=2) + np.sum(np.square(x_err), axis=2) / (2.0 * sigma_x * sigma_x))

reconstruction_errors = np.load('reconstruction_errors.npy')
optimal_latents = np.load('estimated_posterior_latents.npy')
labels = np.load('labels.npy')

log_probs = calc_log_prob(reconstruction_errors, optimal_latents)
predictions = np.argmax(log_probs, axis=1)
assert np.all(np.max(labels, axis=1) == np.min(labels, axis=1))

print predictions[:190]
print labels[:190, 0]
accuracy = (np.sum(predictions[:190] == labels[:190, 0]))/190.0
print accuracy
