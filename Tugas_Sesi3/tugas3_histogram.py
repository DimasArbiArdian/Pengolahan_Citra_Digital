import imageio as img
import numpy as np
import matplotlib.pyplot as plt

path = "C:\\jihyo-twice.jpeg"  
img_rgb = img.imread(path)

# Konversi ke grayscale
img_gray = np.dot (img_rgb[..., :3], [0.2989, 0.5870, 0.1140]) 

# Menghitung histogran dari gambar grayscale
hist_gray, bin_edges = np.histogram(img_gray.flatten(), bins=256, range=(0, 255))

total_pixels = np.sum(hist_gray)

dominant_intensity = np.argmax(hist_gray)
dominant_count = hist_gray[dominant_intensity]

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_gray, cmap='gray')
plt.axis('off')
plt.title("Gambar Grayscale")

plt.subplot(1, 2, 2)
plt.title("Histogram Gambar Grayscale")
plt.xlabel("Intensitas Piksel (0-255)")
plt.ylabel("Frekuensi")
plt.xlim([0, 255])
plt.bar(bin_edges[0:-1], hist_gray, width=1, color='black')

plt.tight_layout()
plt.show()
# Menampilkan jumlah total piksel untuk setiap intensitas
for i in range(256):
    print(f"Intensitas {i}: {hist_gray[i]} piksel")

# Menentukan intensitas dominan
dominant_intensity = np.argmax(hist_gray)

print(f"Total jumlah piksel: {total_pixels}")
print(f"Intensitas dominan: {dominant_intensity} dengan jumlah piksel: {dominant_count}")