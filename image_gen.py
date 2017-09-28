import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
import pickle
from tracker import tracker

# read in saved object and image points
dist_pickle = pickle.load(open("calibration_pickle_new.p", "rb"))
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']

def abs_sobel_thresh(img, sobel_kernel, orient='x', thresh=(0, 255)):
        
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel>= thresh[0]) & (scaled_sobel<=thresh[1])] = 1
            # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image
    return sxbinary

def mag_thresh(img, sobel_kernel, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_mag = np.zeros_like(gradmag)
    binary_mag[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_mag


def dir_threshold(img, sobel_kernel, thresh=(0, np.pi/2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F,1,0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F,0,1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_dir = np.zeros_like(grad_dir)
    binary_dir[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_dir

def color_threshold(img, sthreshold = (0,255), vthreshold = (0,255)):
	#converting image to HLS
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

	#selecting the saturation channel
	s_channel = hls[:,:,2]

	# creating a binary mask where color thresholds are met
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= sthreshold[0]) & (s_channel <= sthreshold[1])] = 1

	#converting image to HSV
	hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

	#selecting the value channel
	v_channel = hsv[:,:,2]

	# creating a binary mask where color thresholds are met
	v_binary = np.zeros_like(v_channel)
	v_binary[(v_channel >= vthreshold[0]) & (v_channel <= vthreshold[1])] = 1

	binary_output = np.zeros_like(s_channel)
	binary_output[(s_binary == 1) & (v_binary == 1)] = 1
	return binary_output

def window_mask(width, height, img_ref, center, level):
	output = np.zeros_like(img_ref)
	output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height), max(0,int(center-width)):min(int(center+width),img_ref.shape[1])] = 1
	return output

# list of test imagesl
images = glob.glob('./test_images/test*.jpg')


# set kernel size
kernel_size = 3


for idx, fname in enumerate(images):
	#read in image
	img = cv2.imread(fname)

	#undistort image
	img = cv2.undistort(img, mtx, dist, None, mtx)

	# process image and generate binary pixel mask
	preprocessImage = np.zeros_like(img[:,:,0])
	gradientx = abs_sobel_thresh(img, sobel_kernel=kernel_size, orient='x', thresh=(50, 255))
	gradienty = abs_sobel_thresh(img, sobel_kernel=kernel_size, orient='y', thresh=(50, 255))
	mag_binary = mag_thresh(img, sobel_kernel=kernel_size, mag_thresh=(30, 100))
	dir_binary = dir_threshold(img, sobel_kernel=kernel_size, thresh=(0, np.pi/2))
	color_binary = color_threshold(img, sthreshold = (100,255), vthreshold = (50,255))
	preprocessImage[((gradientx == 1) & (gradienty == 1) | (color_binary == 1) & (dir_binary==1))] = 255

	# writing binary images to folder
	result = preprocessImage
	write_binary = './test_images/binary/binary'+str(idx)+'.jpg'
	cv2.imwrite(write_binary, result)

	# area where we want to focus on for the perspective transform
	img_size = (img.shape[1], img.shape[0])
	offset = 200 

	# setting our source points
	area_of_interest = [[580,460],[710,460],[1150,720],[150,720]]
	src = np.float32(area_of_interest)
	
	# set our destination points
	dst = np.float32([[offset, 0], [img_size[0]-offset, 0], [img_size[0]-offset, img_size[1]], [offset, img_size[1]]])
	
	# make the transform
	M = cv2.getPerspectiveTransform(src, dst)
	Minv = cv2.getPerspectiveTransform(dst, src)
	binary_warped = cv2.warpPerspective(preprocessImage, M, img_size, flags=cv2.INTER_LINEAR)

	result1 = binary_warped
	write_warped = './test_images/warped/warped'+str(idx)+'.jpg'
	cv2.imwrite(write_warped, result1)

	window_width=25
	window_height=80

	# set up overall class to do all the tracking
	curve_centers = tracker(myWindow_width=window_width, myWindow_height=window_height, myMargin=25, my_ym=10/720, my_xm=4/384, mySmooth_factor=15)

	window_centroids = curve_centers.find_window_centroids(result1)

	# Points used to draw all the left and right windows
	l_points = np.zeros_like(result1)
	r_points = np.zeros_like(result1)

	# points used to find left and right lanes
	rightx = []
	leftx = []

	# Go through each level and draw the windows 	
	for level in range(0,len(window_centroids)):
		# Window_mask is a function to draw window areas
		leftx.append(window_centroids[level][0])
		rightx.append(window_centroids[level][1])
		l_mask = window_mask(window_width,window_height,result1,window_centroids[level][0],level)
		r_mask = window_mask(window_width,window_height,result1,window_centroids[level][1],level)
		# Add graphic points from window mask here to total pixels found 
		l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
		r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

	# Draw the results
	template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
	zero_channel = np.zeros_like(template) # create a zero color channle 
	template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
	warpage = np.array(cv2.merge((result1,result1,result1)),np.uint8) # making the original road pixels 3 color channels
	result2 = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results

	# write images to folder
	write_convolution = './test_images/convolution/conv'+str(idx)+'.jpg'
	cv2.imwrite(write_convolution, result2)
	

	# fit the lane boundaries to the left,right center positions found
	yvals = range(0, result2.shape[0])

	res_yvals = np.arange(result2.shape[0]-(window_height/2),0,-window_height)

	left_fit = np.polyfit(res_yvals, leftx, 2)
	left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
	left_fitx = np.array(left_fitx, np.int32)

	right_fit = np.polyfit(res_yvals, rightx, 2)
	right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
	right_fitx = np.array(right_fitx, np.int32)	

	# make the x and y values fitted to the picture looks slightly nicer
	left_lane = np.array(list(zip(np.concatenate((left_fitx - window_width/2, left_fitx[::-1]+window_width/2), axis=0), np.concatenate((yvals,yvals[::-1]), axis=0))), np.int32)
	right_lane = np.array(list(zip(np.concatenate((right_fitx - window_width/2, right_fitx[::-1]+window_width/2), axis=0), np.concatenate((yvals,yvals[::-1]), axis=0))), np.int32)
	middle_marker = np.array(list(zip(np.concatenate((right_fitx - window_width/2, right_fitx[::-1]+window_width/2), axis=0), np.concatenate((yvals,yvals[::-1]), axis=0))), np.int32)

	# draw colored line on each lane
	road = np.zeros_like(img)
	road_bkg = np.zeros_like(img)
	cv2.fillPoly(road, [left_lane], color=[255,0 ,0])
	cv2.fillPoly(road, [right_lane], color=[0, 0 , 255])
	cv2.fillPoly(road_bkg, [left_lane], color=[255, 255 , 255])
	cv2.fillPoly(road_bkg, [right_lane], color=[255, 255 , 255])

	result3 = road

	write_binary_fit = './test_images/binary_fit/binary_fit'+str(idx)+'.jpg'
	cv2.imwrite(write_binary_fit, result3)

	# use the inverse of matrix M defined earlier to put our image back to normal
	road_warped = cv2.warpPerspective(road, Minv, img_size, flags=cv2.INTER_LINEAR)
	road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, img_size, flags=cv2.INTER_LINEAR)

	lane_color = cv2.addWeighted(img, 1.0, road_warped_bkg, -1.0, 0.0)
	result4 = cv2.addWeighted(lane_color, 1.0, road_warped, 1.0, 0.0)


	write_warped_back = './test_images/warped_back/warped_back'+str(idx)+'.jpg'
	cv2.imwrite(write_warped_back, result4)

	# meters per pixel in y and x dimension
	ym_per_pix = curve_centers.ym_per_pix
	xm_per_pix = curve_centers.xm_per_pix

	# fit the radius of the curve
	curve_fit = np.polyfit(np.array(res_yvals, np.float32)*ym_per_pix, np.array(leftx,np.float32)*xm_per_pix, 2)
	curverad = ((1 + (2*curve_fit[0]*yvals[-1]*ym_per_pix + curve_fit[1])**2)**1.5) / np.absolute(2*curve_fit[0])

	# calculate offset of car from center of the road
	camera_center = (left_fitx[-1] + right_fitx[-1])/2
	center_difference = (camera_center - result1.shape[1]/2)*xm_per_pix
	side_pos = 'left'
	if center_difference <=0:
		side_pos = 'right'

	# draw the text on the screen
	cv2.putText(result4, 'Radius of Curvature = ' + str(round(curverad, 3)) + '(m)', (50,50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
	cv2.putText(result4, 'Vehicle is ' + str(abs(round(center_difference, 3))) + 'm ' +side_pos+ ' of center', (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

	write_text = './test_images/write_text/write_text'+str(idx)+'.jpg'
	cv2.imwrite(write_text, result4)