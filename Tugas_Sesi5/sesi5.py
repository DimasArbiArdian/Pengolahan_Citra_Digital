import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
color_image = cv2.imread('jihyo-twice.jpeg')
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

# Low-pass filter (smoothing)
low_pass_kernel = np.ones((5, 5), np.float32) / 25
low_pass_color = cv2.filter2D(color_image, -1, low_pass_kernel)
low_pass_gray = cv2.filter2D(gray_image, -1, low_pass_kernel)

# High-pass filter (edge detection)
high_pass_kernel = np.array([[-1, -1, -1],
                             [-1,  8, -1],
                             [-1, -1, -1]])
high_pass_color = cv2.filter2D(color_image, -1, high_pass_kernel)
high_pass_gray = cv2.filter2D(gray_image, -1, high_pass_kernel)

# High-boost filter (original + edge)
k = 1.5  # Boost factor
high_boost_gray = gray_image + k * high_pass_gray
high_boost_color = cv2.addWeighted(color_image, 1, high_pass_color, k, 0)

# Display results
titles = ['Original Color', 'Low-pass Color', 'High-pass Color', 'High-boost Color',
          'Original Grayscale', 'Low-pass Grayscale', 'High-pass Grayscale', 'High-boost Grayscale']
images = [color_image, low_pass_color, high_pass_color, high_boost_color,
          gray_image, low_pass_gray, high_pass_gray, high_boost_gray]

plt.figure(figsize=(12, 8))
for i in range(8):
    plt.subplot(2, 4, i+1)
    if len(images[i].shape) == 3:  # Color image
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    else:  # Grayscale image
        plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
