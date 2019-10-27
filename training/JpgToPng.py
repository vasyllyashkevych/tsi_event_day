import cv2
import glob

base_path = "ds-vasyl-lyashkevych/places/"
images = []

for filename in glob.glob(base_path + '*.jpg'):
    images.append(filename)


for filename in glob.glob(base_path + '*.png'):
    images.append(filename)


index = 1
for i in images:
    im = cv2.imread(i)
    cv2.imwrite(base_path + str(index)+".png", im)
    index += 1

print(len(images))
