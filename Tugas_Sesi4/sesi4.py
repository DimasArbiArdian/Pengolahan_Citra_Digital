import imageio as img
import numpy as np
from scipy.ndimage import histogram

def histogram_equalization(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    equalized_image = cdf[image]
    return equalized_image

input_image = img.imread('z.jpg')  
if input_image.ndim == 3:  
    input_image = np.mean(input_image, axis=2).astype(np.uint8)  

output_image = histogram_equalization(input_image)

img.imwrite('output.jpg', output_image)