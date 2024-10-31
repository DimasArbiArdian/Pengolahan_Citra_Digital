import numpy as np
import imageio as img
import matplotlib.pyplot as plt

path = 'C://jihyo.jpg'
image = img.imread(path)

height, width = image.shape[:2]
horizontal = np.zeros_like(image)
vertical = np.zeros_like(image)
horizontal_vertical_mirror = np.zeros_like(image)

for y in range(height):
    for x in range(width):
        horizontal[y,x] = image[y, width - 1 - x]

for y in range(height):
    for x in range(width):
        vertical[y,x] = image[height - 1-y,x]

plt.figure(figsize=(10,5))

plt.subplot(1, 4, 1)
plt.imshow(image)

plt.subplot(1, 4, 2)
plt.imshow(horizontal)

plt.subplot(1, 4, 3)
plt.imshow(vertical)

plt.show()