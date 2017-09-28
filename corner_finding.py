import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob

objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('./camera_cal/calibration*.jpg')

def corner_finding(img, nx, ny):
	# Step through the list and search for chessboard corners
	for idx, fname in enumerate(img):
		img = cv2.imread(fname)
		img_size = img_size = (img.shape[1], img.shape[0])
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	    # Find the chessboard corners
		ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

	    # If found, add object points, image points
		if ret == True:
			print('working on ', fname)
			objpoints.append(objp)
			imgpoints.append(corners)

			# Draw the corners and write to a file
			img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
			ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
			write_name = 'corners_found_NEW'+str(idx)+'.jpg'
			cv2.imwrite(write_name, img)

	
	# Save the camera calibration results for later use.
	dist_pickle = {}
	dist_pickle['objpoints'] = objpoints
	dist_pickle['imgpoints'] = imgpoints
	dist_pickle['mtx'] = mtx
	dist_pickle['dist'] = dist
	pickle.dump(dist_pickle, open('./calibration_pickle_new.p', 'wb'))

corner_finding(images, 9, 6)



