import os
import numpy as np
from matplotlib import pyplot as plt
from cv2 import imwrite

real_digits = np.load('real_digits.npy')
recons_digits = np.load('reconstructed_digits.npy')
found_latents = np.load('estimated_posterior_latents.npy')
recons_from_noisy_latents = np.load('reconstructed_digits_with_noisy_latents.npy')

k, n, d = recons_digits.shape
w = 28
assert d == w*w

mosaic = np.zeros([w*(k+1)+1, w*(n+1)])
mosaic[w, :] = 1
for i in range(n):
    mosaic[:w, i*w:(i+1)*w] = real_digits[i].reshape([w, w])
    for j in range(k):
        mosaic[(j+1)*w+1:(j+2)*w+1, i*w:(i+1)*w] = recons_digits[j, i].reshape([w, w])

imwrite('per_class_reconstructions.png', mosaic*255)

# Show noisy-z reconstructions
r = recons_from_noisy_latents.shape[1]
mosaic = np.zeros([w*(k+1)+1, w*(r+1)+1])
mosaic[w, :] = 1
mosaic[:, w] = 1
i = 0
mosaic[:w, :w] = real_digits[i].reshape([w, w])
for j in range(k):
    mosaic[(j+1)*w+1:(j+2)*w+1, :w] = recons_digits[j, i].reshape([w, w])
    for l in range(r):
        mosaic[(j+1)*w+1:(j+2)*w+1, (l+1)*w+1:(l+2)*w+1] = recons_from_noisy_latents[j, l, i].reshape([w, w])
imwrite('per_class_reconstructions_from_noisy_z_for_sample_{}.png'.format(i), mosaic*255)

z_ll = -k*np.log(2*np.pi) - 0.5 * np.sum(np.square(found_latents), axis=2)
recons_loss = np.sum(np.abs(real_digits-recons_digits), axis=2)
print z_ll.shape

for i in range(8):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(z_ll[:, i], '.-')
    plt.grid(True)
    plt.ylabel('log(P(z))')
    plt.title('Sample number {}'.format(i+1))
    plt.subplot(2, 1, 2)
    plt.plot(recons_loss[:, i], '.-')
    plt.ylabel('|Real - G(z))')
    plt.xlabel('Model')
    plt.grid(True)
    plt.savefig('ll_per_model_for_sample_num_{}.png'.format(i+1))
plt.show()
