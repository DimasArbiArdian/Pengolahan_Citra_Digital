import numpy as np
import matplotlib.pyplot as plt
import imageio as img

def zoomMinus(image, factor):
    height, width = image.shape[:2]
    new_height = int(height / factor)
    new_width = int(width / factor)
    imgZoom = np.zeros((new_height, new_width, 3), dtype=image.dtype)
    
    for y in range(new_height):
        for x in range(new_width):
            ori_y = int(y * factor)
            ori_x = int(x * factor)
            
            ori_y = min(ori_y, height - 1)
            ori_x = min(ori_x, width - 1)
            
            imgZoom[y, x] = image[ori_y, ori_x]
    
    return imgZoom

image = img.imread('C:\\jihyo.jpg')

skala = 2.0  
imgZoom = zoomMinus(image, skala)

img.imwrite("z_minus.jpg", imgZoom)

plt.subplot(1, 2, 1)
plt.title("Gambar Asli")
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title("Gambar Diperkecil")
plt.imshow(imgZoom)

plt.show()
