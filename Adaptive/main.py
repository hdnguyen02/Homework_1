import cv2
import matplotlib.pyplot as plt

dental_image = cv2.imread('dental.jpg', 0)
parrot_image = cv2.imread('parrot.jpg', 0)
skull_image = cv2.imread('skull.jpg', 0)

clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))

ahe_dental_image = clahe.apply(dental_image)
ahe_parrot_image = clahe.apply(parrot_image)
ahe_skull_image = clahe.apply(skull_image)

plt.figure(figsize=(12, 9))

# Dental
plt.subplot(3, 4, 1)
plt.title('Original Dental Image')
plt.imshow(dental_image, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 2)
plt.title('AHE Dental Image')
plt.imshow(ahe_dental_image, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 3)
histogram_dental_origin = cv2.calcHist([dental_image], [0], None, [256], [0, 256])
plt.title('Original Dental Histogram')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(histogram_dental_origin)
plt.xlim([0, 256])

plt.subplot(3, 4, 4)
histogram_dental_ahe = cv2.calcHist([ahe_dental_image], [0], None, [256], [0, 256])
plt.title('AHE Dental Histogram')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(histogram_dental_ahe)
plt.xlim([0, 256])

# Parrot

plt.subplot(3, 4, 5)
plt.title('Original Parrot Image')
plt.imshow(parrot_image, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 6)
plt.title('AHE Parrot Image')
plt.imshow(ahe_parrot_image, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 7)
histogram_parrot_origin = cv2.calcHist([parrot_image], [0], None, [256], [0, 256])
plt.title('Original Parrot Histogram')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(histogram_parrot_origin)
plt.xlim([0, 256])

plt.subplot(3, 4, 8)
histogram_parrot_ahe = cv2.calcHist([ahe_parrot_image], [0], None, [256], [0, 256])
plt.title('AHE Parrot Histogram')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(histogram_parrot_ahe)
plt.xlim([0, 256])


# Skull

plt.subplot(3, 4, 9)
plt.title('Original Skull Image')
plt.imshow(skull_image, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 10)
plt.title('AHE Skull Image')
plt.imshow(ahe_skull_image, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 11)
histogram_skull_origin = cv2.calcHist([skull_image], [0], None, [256], [0, 256])
plt.title('Original Skull Histogram')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(histogram_skull_origin)
plt.xlim([0, 256])

plt.subplot(3, 4, 12)
histogram_skull_ahe = cv2.calcHist([ahe_skull_image], [0], None, [256], [0, 256])
plt.title('AHE Skull Histogram')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(histogram_skull_ahe)
plt.xlim([0, 256])

plt.tight_layout()
plt.show()
