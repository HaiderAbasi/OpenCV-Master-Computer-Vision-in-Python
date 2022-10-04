import cv2
import numpy as np
from loguru import logger
import time
from src.a__IP_Basics.utilities import build_montages,print_h,describe,putText

def estimate_drone_pose(drone_view,map):
	#  Hint     : Map was created with drone at nadir angle, drone pase is unknown in drone view so it needs to be estimated
	#  Hint     : Find Homography will be be critical here. (Both forward and in inverse)
	rel_location = None
	rel_orientation = None
	# Write Code here


	return rel_location,rel_orientation




def assignment():
	# Assignment: Drone after its long autonomous flight on its way to the landing pad suddenly lost
	#             its GPS-signal and odometry data was always buggy to begin with,
	#             Luckily the area was mapped out by the drone so at any point where the drone had these
	#             malfunctions, it should be inside the mapped zone. This means you can leverage this information
	#             for fetching the drone relative location and pose.
	#  
	#  Task     : Estimate the drone relative location + Pose utilizing the the drone view with known map using feature detection and 
	#             mapping
	print_h("[Assignment]:  Estimate the relative drone pose (utilizing the the drone view with known map using feature detection and mapping)")

	# Input
	drone_view = cv2.imread("Data/NonFree/test\DSC00153_small.JPG")
	images = []
	titles = []
	images.append(drone_view)
	titles.append("drone_view")
	map = cv2.imread("Data/NonFree/test/building_mosaic.tif")


	# Task Function
	rel_location,rel_orientation = estimate_drone_pose(drone_view,map)
	if ((rel_location==None) or (rel_orientation==None)):
		logger.error("estimate_drone_pose() needs to be coded to get the required(rel_location + rel_orientation) result.")
		exit(0)
	else:
		# Valid pose found
		pose_txt = "drone pose = (loc = {rel_location} , orientation = {rel_orientation})"
		putText(map,pose_txt,(20,20))

	# Output (Display)
	images.append(map)
	titles.append("map")
	montage = build_montages(images,None,None,titles,True,True)
	for montage_img in montage:
		cv2.imshow("Pose-estimation",montage_img)
	cv2.waitKey(0)




def vis_features(img):
	# SURF and SIFT are very robust, and perform well under scale and rotation variances. Affine shifts are a little tricky, but not bad. And FAST is not a descriptor, it is just a (mind-boggling fast!) detector.
	# If you're considering eligibility for real-time tests, then I'm afraid you'll have to trade-off a great deal of performance. SIFT and SURF are not real-time. Others are relatively faster (BRISK should top it, if I recall)
	images = []
	titles = []

	images.append(img)
	titles.append("deans_car")

	# A) Harris Corner detector : The first keypoint detector available in OpencV
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # grayscale and float32 type.
	
	img_sift = img.copy()
	sift = cv2.SIFT_create()
	
	kp  = sift.detect(gray,None)
	cv2.drawKeypoints(img,kp,img_sift,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS|cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
	[descriptors, keypoints] = sift.compute(img,kp)
	
	images.append(img_sift)
	titles.append("Sift")

	img_orb = img.copy()
	# Initiate ORB detector
	# scaleFactor	Pyramid decimation ratio
	# nlevels	The number of pyramid levels.
	# nfeatures	The maximum number of features to retain.    
	orb = cv2.ORB_create()
	# find the keypoints and descriptors with ORB
	kp1, des1 = orb.detectAndCompute(img,None)
	# draw only keypoints location,not size and orientation
	cv2.drawKeypoints(img, kp1, img_orb, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS|cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
	images.append(img_orb)
	titles.append("ORB")

	# Displaying image and threshold result
	montage = build_montages(images,None,None,titles,True,True)
	for montage_img in montage: 
		#imshow("Found Clusters",cluster,cv2.WINDOW_AUTOSIZE)
		cv2.imshow("Features",montage_img)
	cv2.waitKey(0)


def vis_keypoints(img):
	images = []
	titles = []

	images.append(img)
	titles.append("deans_car")

	# A) Harris Corner detector : The first keypoint detector available in OpencV
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # grayscale and float32 type.
	blockSize = 2 # size of neighbourhood considered for corner detection
	ksize = 7     # Aperture parameter of the Sobel derivative used.
	k = 0.08      # Harris detector free parameter in the equation. 
	dst = cv2.cornerHarris(gray,blockSize,ksize,k) # dst [Float_32 size same as src]
	dst = cv2.dilate(dst,None)
	# Threshold for an optimal value, it may vary depending on the image.
	img_harris = img.copy()
	img_harris[dst>0.01*dst.max()]=[0,0,255]
	images.append(img_harris)
	titles.append("Corners (Harris)")

	# B) Shi-Thomsi Corner detector (Important: Improved Harris Corner so better to use this!)
	img_shi_thomsi = img.copy()
	max_corners = 25   # Maximum numbers of corner you wish to get
	min_quality = 0.01 # Minimum quality required to be considered a valid corner Range (0-1)
	min_euc_dist = 10  # Minimum allowed euc distance between two corners
	corners = cv2.goodFeaturesToTrack(gray,max_corners,min_quality,min_euc_dist) # Returns detected corners (Float)
	corners = np.int0(corners) # Convert corners to int for display
	for i in corners:
		x,y = i.ravel() # unpack
		img_shi_thomsi = cv2.circle(img_shi_thomsi,(x,y),8,(255,0,0),-1)
	images.append(img_shi_thomsi)
	titles.append("Corners (Shi-Thomsi)")

	# Displaying image and threshold result
	montage = build_montages(images,None,None,titles,True,True)
	for montage_img in montage:
		#imshow("Found Clusters",cluster,cv2.WINDOW_AUTOSIZE)
		cv2.imshow("Keypoints",montage_img)
	cv2.waitKey(0)


def find_obj_inscene(obj_,scene_,method="sift",benchmarking = False,resize=1):
	if benchmarking:
		start = time.time()
	
	obj = obj_.copy()
	scene = scene_.copy()
	if resize!=1:
		obj = cv2.resize(obj,None,fx=resize,fy=resize)
		scene = cv2.resize(scene,None,fx=resize,fy=resize)


	images = []
	titles = []
	MIN_MATCH_COUNT = 10

	# Initiate ORB detector
	if method =="orb":
		print("\n> Using ORB for detection and matching")
		orb = cv2.ORB_create()
		# find the keypoints and descriptors with ORB
		kp1, des1 = orb.detectAndCompute(obj,None)
		kp2, des2 = orb.detectAndCompute(scene,None)
		# create BFMatcher object
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
		# Match descriptors.
		matches = bf.match(des1,des2)
		# Sort them in the order of their distance.
		matches = sorted(matches, key = lambda x:x.distance)
		# Draw first 10 matches.
		good = matches[:20]
	elif method =="sift":
		print("\n> Using Sift for detection and matching")
		sift = cv2.SIFT_create()
		# find the keypoints and descriptors with SIFT
		kp1, des1 = sift.detectAndCompute(obj,None)
		kp2, des2 = sift.detectAndCompute(scene,None)
		# BFMatcher with default params
		bf = cv2.BFMatcher()
		matches = bf.knnMatch(des1,des2,k=2)
		# Apply ratio test
		good = []
		for m,n in matches:
			if m.distance < 0.75*n.distance:
				good.append(m)
	elif method =="sift-flann":
		print("\n> Using Sift-flann for detection and matching")
		# Initiate SIFT detector
		sift = cv2.SIFT_create()
		# find the keypoints and descriptors with SIFT
		kp1, des1 = sift.detectAndCompute(obj,None)
		kp2, des2 = sift.detectAndCompute(scene,None)
		# FLANN parameters
		FLANN_INDEX_KDTREE = 1
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks=50)   # or pass empty dictionary
		flann = cv2.FlannBasedMatcher(index_params,search_params)
		matches = flann.knnMatch(des1,des2,k=2)
		# store all the good matches as per Lowe's ratio test.
		good = []
		for m,n in matches:
			if m.distance < 0.7*n.distance:
				good.append(m)
	else:
		print(f"Unknown method specified = {method}")
		return

	
	if len(good)>MIN_MATCH_COUNT:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
		matchesMask = mask.ravel().tolist()
		h,w,d = obj.shape
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		dst = cv2.perspectiveTransform(pts,M)
		scene = cv2.polylines(scene,[np.int32(dst)],True,255,3, cv2.LINE_AA)
	else:
		print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
		matchesMask = None

	
	if benchmarking:
		print(f"Total time taken to findobjinscene using ORB = {time.time()-start} ms")
	else:
		draw_params = dict(matchColor = (0,255,0), # draw matches in green color
						singlePointColor = None,
						matchesMask = matchesMask, # draw only inliers
						flags = 2)
		matched_img = cv2.drawMatches(obj,kp1,scene,kp2,good,None,**draw_params)
		
		
		images.append(matched_img)
		titles.append(f"Feature Matching ({method})")


		# Displaying image and threshold result
		montage = build_montages(images,None,None,titles,True,True)
		for montage_img in montage:
			method_str = f"({method})"
			cv2.imshow("find_Obj_in_Scene " + method_str.upper(),montage_img)
		cv2.waitKey(0)



def main():
	print_h("[main]: OpenCV Image feature-extraction and Usage.")

	# Read the image that we would use for feature detection
	img = cv2.imread("Data\CV\supernatural-impala.jpg")

	print_h("[a]: Extracting and Visualizing keypoints and features of an image.")
	vis_keypoints(img)
	vis_features(img)
	cv2.destroyAllWindows()

	print_h("[b]: Using feature-matching to find obj in scene.")
	# Application: Finding known object in a scene
	obj = cv2.imread("Data\CV\ltp.jpg")
	scene = cv2.imread("Data\CV\scene2.jpg")

	find_obj_inscene(obj,scene,"orb") 

	find_obj_inscene(obj,scene,"sift") 
	
	find_obj_inscene(obj,scene,"sift-flann")
	cv2.destroyAllWindows()



if __name__ =="__main__":
	main()

	assignment()