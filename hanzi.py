import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt     # Matplotlib的pyplot模块，用于图像的显示和绘制

# 读取灰度图像
image = cv.imread('hanzi1.jpg', 0)
img = cv.imread('hanzi1.jpg')

# 裁剪图片
x, y, w, h = 100, 530, 2900, 3300
cropped_image = image[y:y+h, x:x+w]

x, y, w, h = 100, 530, 2900, 3300
cropped_img = img[y:y+h, x:x+w]

# 设置阈值
ret, thresh = cv.threshold(cropped_image, 127, 255, cv.THRESH_BINARY_INV)
# 将图像转换为二值图像
_,binary_image = cv.threshold(thresh, ret, 255, cv.THRESH_BINARY)
image_gb = cv.medianBlur(binary_image, 5) 

element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))  # 定义一个3x3的十字形结构元素

fushi_image = cv.erode(binary_image, element, iterations = 4) 
image_gb1 = cv.medianBlur(fushi_image, 5) 

pengzhang_image = cv.dilate(image_gb1, element, iterations = 6) 
image_gb2 = cv.medianBlur(pengzhang_image, 5) 

kernel = np.ones((9, 9), np.uint8)
closed_img = cv.morphologyEx(image_gb2, cv.MORPH_CLOSE, kernel, iterations = 10)
 
lower = 50
upper = 200
img_blur = cv.GaussianBlur(closed_img,(3,3), 0) #高斯滤波
edges_with_blur = cv.Canny(img_blur, lower, upper)

contours, _ = cv.findContours(edges_with_blur, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

img_copy = cropped_img.copy()

for c in contours:
    perimeter = cv.arcLength(c, True)

    x, y, w, h = cv.boundingRect(c)

    # 根据周长阈值筛选并绘制边界框
    if perimeter > 100:
        cv.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)


# 使用matplotlib显示原始图像和滤波后的图像
fig, ax = plt.subplots(1, 7, figsize=(20, 10))
ax[0].imshow(cropped_image, cmap='gray')
ax[0].axis('off')
# ax[0].set_title("Grayscale image", fontsize=16)
ax[1].imshow(binary_image, cmap='gray')
ax[1].axis('off')
# ax[1].set_title("Global Thresholding >> Binarized Image", fontsize=16)
ax[2].imshow(fushi_image, cmap='gray')
ax[2].axis('off')
# ax[2].set_title("Apply Erosion Operation >> Remove Noise", fontsize=16)
ax[3].imshow(pengzhang_image, cmap='gray')
ax[3].axis('off')
# ax[3].set_title("Apply Dilation Operation >> Enhance Image Features >> Median Filtering to Remove Small White Dots", fontsize=16)
ax[4].imshow(closed_img, cmap='gray')
ax[4].axis('off')
# ax[4].set_title("Apply Closing Operation >> Fill Closed Regions", fontsize=16)
ax[5].imshow(edges_with_blur, cmap='gray')
ax[5].axis('off')
# ax[5].set_title("Canny edge detection", fontsize=16)
ax[6].imshow(img_copy)
ax[6].axis('off')
# ax[6].set_title("recognition result", fontsize=16)

plt.tight_layout()
plt.show()


# 保存灰度图像（如果需要）
cv.imwrite('gray_image.jpg', img_copy)
