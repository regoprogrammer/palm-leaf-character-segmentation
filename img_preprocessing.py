import cv2

from skimage.restoration import denoise_tv_chambolle

# Load the image
img = cv2.imread("test1.jpg")
# Display the image in a window
cv2.imshow("image", img)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Save the grayscale image
cv2.imwrite("output.jpg", gray_img)

# Display the grayscale image
cv2.imshow("Grayscale Image", gray_img)
cv2.waitKey(0)

#noice reduction
blurred_image = cv2.GaussianBlur(gray_img, (5, 5), 0)

# Display the original and blurred images
cv2.imshow("Original Image", gray_img)
cv2.waitKey(0)
cv2.imshow("Blurred Image", blurred_image)
cv2.waitKey(0)
equalized_image = cv2.equalizeHist(gray_img)
denoised_image = denoise_tv_chambolle(equalized_image, weight=0.1)
# Save the enhanced image (optional)
cv2.imwrite("enhanced_image.jpg", equalized_image)

cv2.imshow("denoised img",denoised_image)
cv2.waitKey(0)

# Display the original and enhanced images
cv2.imshow("Enhanced Image", equalized_image)

# Apply binary thresholding
# Choose an appropriate threshold value based on your image
threshold_value = 128
_, thresholded_image = cv2.threshold(equalized_image, threshold_value, 255, cv2.THRESH_BINARY)

# Save the thresholded image (optional)
cv2.imwrite("thresholded_image.jpg", thresholded_image)

# Display the original and thresholded images
cv2.imshow("Thresholded Image", thresholded_image)
cv2.waitKey(0)

# Apply Canny edge detection
edges = cv2.Canny(thresholded_image, 100, 200)  # Adjust the threshold values as needed

# Save the edge-detected image (optional)
cv2.imwrite("edges_image.jpg", edges)

# Display the edge-detected images

cv2.imshow("Edge-Detected Image", edges)

# Wait for a key event and close OpenCV windows
cv2.waitKey(0)
cv2.destroyAllWindows()

