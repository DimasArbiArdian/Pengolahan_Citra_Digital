import imageio as img
import numpy as np
import matplotlib.pyplot as plt

image = img.imread('eunbi.jpg', mode = 'F')
sobelX = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
     ])

sobelY = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])

imgPad = np.pad(image, pad_width=1, mode='constant', constant_values=0)

Gx = np.zeros_like(image)
Gy = np.zeros_like(image)

for y in range(1,imgPad.shape[0]-1):
    for x in range(1,imgPad.shape[1]-1):
        region = imgPad[y-1:y+2, x-1:x+2]
        Gx[y-1,x-1] = (region * sobelX).sum()
        Gy[y-1,x-1] = (region * sobelY).sum()

sobel_edge = np.sqrt(Gx**2 + Gy**2)
sobel_edge = (sobel_edge/sobel_edge.max()) * 255
sobel_edge = np.clip(sobel_edge, 0, 255)
sobel_edge = sobel_edge.astype(np.uint8)

def basicThres(sobel_edge,level):
    threshold= np.where(sobel_edge>level,255,0)
    return threshold.astype(np.uint8)

basic_image = basicThres(image, level=128)

plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.title('Gambar Asli')
plt.imshow(image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Tepi Sobel')
plt.imshow(sobel_edge, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Basic Thresholding")
plt.imshow(basic_image, cmap='gray')
plt.show()