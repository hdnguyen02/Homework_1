import cv2
import matplotlib.pyplot as plt

image = cv2.imread('moon.jpg', 0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

clahe_image = clahe.apply(image)

plt.figure(figsize=(12, 4))
plt.subplot(141)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(142)
plt.title('CLAHE Image')
plt.imshow(clahe_image, cmap='gray')
plt.axis('off')

plt.subplot(143)
histogram_origin = cv2.calcHist([image],[0],None,[256],[0,256])
plt.title('Original Histogram')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(histogram_origin)
plt.xlim([0, 256])

plt.subplot(144)
histogram_CLAHE = cv2.calcHist([clahe_image],[0],None,[256],[0,256])
plt.title('CLAHE Histogram')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(histogram_CLAHE)
plt.xlim([0, 256])

plt.tight_layout()
plt.show()
