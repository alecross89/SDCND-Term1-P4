import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob

# load the image and object points
dist_pickle = pickle.load(open("./calibration_pickle_new.p", "rb"))
objpoints = dist_pickle['objpoints']
imgpoints = dist_pickle['imgpoints']
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']

# defining undistort function
def undistort(img, objpoints, imgpoints, img_size, mtx, dist):
    
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    return dst

# load chessboard calibration image for reference
img = cv2.imread('./camera_cal/calibration2.jpg')
img_size = (img.shape[1], img.shape[0])

# example of undistorting a chess board image
undistorted_img_chess = undistort(img, objpoints, imgpoints, img_size, mtx, dist)

# Uncomment below to view images.
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(img)
# ax1.set_title('Original Image', fontsize=20)
# ax2.imshow(undistorted_img_chess)
# ax2.set_title('Undistorted Image', fontsize=20)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# plt.show()


####################################################################

# Example of showing an undistorted vs distorted test image side by side
# load test image for reference
img1 = cv2.imread('./test_images/test1.jpg')
img_size1 = (img1.shape[1], img1.shape[0])

# calling undistort function
undistorted_test_image1 = undistort(img1, objpoints, imgpoints, img_size1, mtx, dist)

# Uncomment below to view images
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(img1)
# ax1.set_title('Original Image', fontsize=20)
# ax2.imshow(undistorted_test_image1)
# ax2.set_title('Undistorted Image', fontsize=20)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# plt.show()

####################################################################

# Below will undistort all the test images and write them into the test_images folder

# read in all images from the test_image folder
images = glob.glob('./test_images/test*.jpg')

for idx, fname in enumerate(images):
	# read in image
	imgs = cv2.imread(fname)

	# undistort the images
	imgs = cv2.undistort(imgs, mtx, dist, None, mtx)
	result = imgs

	# write images to folder
	write_name = './test_images/undistorted_test_images/undistorted'+str(idx)+'.jpg'
	cv2.imwrite(write_name, result)






