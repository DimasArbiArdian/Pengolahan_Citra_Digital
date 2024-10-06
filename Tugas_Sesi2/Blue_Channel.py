import numpy as np
import imageio.v3 as img

image_path = "D:\OneDrive\Documents\RGB\Tugas_Sesi2\Citra Digital\Daun Singkong.jpeg"
image = img.imread(image_path)

if len (image.shape) < 3 or image.shape[2] != 3:
    print("Format Gambar Harus RGB")
    exit()

red = image[:,:,0]
green = image[:,:,1]
blue = image[:,:,2]

image_blue = np.zeros_like(image)
image_blue[:,:,2] = blue

img.imwrite("blue_singkong.jpg", image_blue)

print("Proses Berhasil")
