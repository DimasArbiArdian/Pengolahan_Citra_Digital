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

gray = 0.299 * red + 0.587 * green + 0.144 * blue

image_gray = np.zeros_like(image)
image_gray[:,:,0] = gray
image_gray[:,:,1] = gray
image_gray[:,:,2] = gray

threshold = 100
image_bw = np.zeros_like(image)

for i in range(0, len(gray)):
    for j in range(0, len(gray[i])):
        if gray[i][j] > threshold:
            image_bw[i][j] = 255
        else:
            image_bw[i][j] = 0


img.imwrite("bw_singkong.jpg", image_bw)

print("Proses Berhasil")
