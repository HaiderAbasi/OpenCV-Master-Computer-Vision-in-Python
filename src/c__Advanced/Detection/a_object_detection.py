import cv2
import numpy as np
import time
import os
import math
from src.a__IP_Basics.utilities import putText,print_h,disp_fps

from loguru import logger


def assignment():
	# Train a yolo-v5 or v7 to a custom dataset and use it for a specific use-case here.
	print_h("[Assignment]:  Train a yolo-v5 or v7 to a custom dataset and use it for a specific use-case here.\n")
	
	# Input Video
	dashcam_street = "Data/NonFree/dash_view.webm"

	# Creating instance of yolo v5 trained by you for obj detection in the given test video
	method = "yolo_v5"
	detection_yolo = Detection(method)
	# Object detection in video
	detection_yolo.detect(vid_path = dashcam_street)


# OpenCV object detectors
# from the dnn module
class Dnn:
	
	def __init__(self,model_path = None, weights_path = None,classes_path = None,conf = 0.75):
		#Reference CV Guide : https://docs.opencv.org/3.4/da/d9d/tutorial_dnn_yolo.html
		if model_path==None:
			model_path = 'Data/NonFree\Dnn\yolov3.cfg'
		if weights_path==None:
			weights_path = 'Data/NonFree\Dnn\yolov3.weights'
		# Give the configuration and weight files for the model and load the network.
		# Reference: https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html
		self.net = cv2.dnn.readNetFromDarknet(model_path, weights_path)
		#self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
		self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

		self.classes_path = 'Data/NonFree/Dnn/coco.names'
		# Load names of classes and get random colors
		if classes_path != None:
			self.classes_path = classes_path
		self.classes = open(self.classes_path).read().strip().split('\n')
		np.random.seed(42)
		self.colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype='uint8')

		# determine the output layer
		ln = self.net.getLayerNames()

		self.ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

		# Setting the initial confidence parameter
		self.conf = conf


	def post_process(self,img, outputs, conf):
		H, W = img.shape[:2]

		boxes = []
		confidences = []
		classIDs = []

		for output in outputs:
			scores = output[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if confidence > conf:
				x, y, w, h = output[:4] * np.array([W, H, W, H])
				p0 = int(x - w//2), int(y - h//2)
				p1 = int(x + w//2), int(y + h//2)
				boxes.append([*p0, int(w), int(h)])
				confidences.append(float(confidence))
				classIDs.append(classID)
				# cv2.rectangle(img, p0, p1, WHITE, 1)

		FONT_SCALE = 2e-3  # Adjust for larger font size in all images
		THICKNESS_SCALE = 1e-3  # Adjust for larger thickness in all images
		# lower the nmsconf --> The more boxes get supressed as 1 #
		indices = cv2.dnn.NMSBoxes(boxes, confidences, conf, conf-0.2)
		if len(indices) > 0:
			for i in indices.flatten():
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
				color = [int(c) for c in self.colors[classIDs[i]]]
				cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
				text = "{}: {:.2f}".format(self.classes[classIDs[i]], confidences[i])

				thickness = math.ceil(min(w, h) * THICKNESS_SCALE)
				#fnt_scale = get_optimal_font_scale(text,w)[0]


				height, width= img.shape[:2]
				font_scale = min(width, height) * FONT_SCALE
				thickness = math.ceil(min(width, height) * THICKNESS_SCALE)
				putText(img, text,  org= (x, y - 15),fontScale=font_scale,color=(255,0,255),thickness=1)

	def trackbar(self,x,img0,outputs):

		img = img0.copy()
		conf = x/100
		
		self.post_process(img, outputs, conf)

		cv2.imshow('window', img)
		
	def detect(self,img0):
		img = img0.copy() # Duplicating input image

		# construct a blob from the image
		blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
		
		self.net.setInput(blob)
		
		t0 = time.time()
		outputs = self.net.forward(self.ln) # Predicting using Yolov3
		t = time.time()

		# combine the 3 output groups into 1 (10647, 85)
		# large objects (507, 85)
		# medium objects (2028, 85)
		# small objects (8112, 85)
		outputs = np.vstack(outputs)

		self.post_process(img, outputs, 0.5)

		cv2.namedWindow('window')
		#cv2.createTrackbar('confidence', 'blob', 50, 101,lambda x: self.trackbar2(x,r0,outputs) )
		cv2.createTrackbar('confidence', 'window', 50, 100, lambda x: self.trackbar(x,img0,outputs))
		# GoodRead https://stackoverflow.com/questions/40680795/cv2-createtrackbar-pass-userdata-parameter-into-callback

		#cv2.displayOverlay('window', f'forward propagation time={t-t0}')
		cv2.putText(img,f'forward propagation time={t-t0}',(50,100),cv2.FONT_HERSHEY_PLAIN,1.5,(255,0,0),1)

		cv2.imshow('window',img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	def detect_in_video(self,vid_path,save_vid = False):

		vid = cv2.VideoCapture(vid_path)

		if save_vid:
			# Default resolutions of the frame are obtained.The default resolutions are system dependent.
			# We convert the resolutions from float to integer.
			frame_width = int(vid.get(3))
			frame_height = int(vid.get(4))
			filename, ext = os.path.splitext(vid_path)
			# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
			out = cv2.VideoWriter(f'{filename}_yolo_V7-tiny.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (frame_width,frame_height))

		while(vid.isOpened()):
			ret,frame = vid.read()
			if ret:
				start_time = time.time()
				# Detect using Yolo

				# construct a blob from the image
				blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
				
				self.net.setInput(blob)
				
				t0 = time.time()
				outputs = self.net.forward(self.ln) # Predicting using Yolov3
				t = time.time()

				# combine the 3 output groups into 1 (10647, 85)
				# large objects (507, 85)
				# medium objects (2028, 85)
				# small objects (8112, 85)
				outputs = np.vstack(outputs)

				self.post_process(frame, outputs, self.conf)

				disp_fps(frame,start_time)
				cv2.imshow('Live Obj-det',frame)
				if save_vid:
					out.write(frame)
				k=cv2.waitKey(1)
				if k==27:
					break
			else:
				if save_vid:
					out.release()
				print("Video Ended")
				break
		if save_vid:
			out.release()

# from the detection module
class CascadeDetector:

	def __init__(self,face_cascade_path = None,eyes_cascade_path = None):

		if face_cascade_path == None:
			face_cascade_path = 'Data/NonFree/haarcascades/haarcascade_frontalface_alt.xml'
		if eyes_cascade_path == None:
			eyes_cascade_path = 'Data/NonFree/haarcascades/haarcascade_eye_tree_eyeglasses.xml'

		self.face_cascade = cv2.CascadeClassifier()
		self.eyes_cascade = cv2.CascadeClassifier()
		#-- 1. Load the cascades
		if not self.face_cascade.load(face_cascade_path):
			logger.error("--(!)Error loading face cascade")
			exit(0)
		if not self.eyes_cascade.load(eyes_cascade_path):
			logger.error("--(!)Error loading eyes cascade")
			exit(0)

	def detectAndDisplay(self,frame,display = False,detect_eyes = False):
		start_time = time.time() # Timing 
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#frame_gray = cv2.equalizeHist(frame_gray)
		#-- Detect faces
		faces = self.face_cascade.detectMultiScale(frame_gray)
		if display:
			for (x,y,w,h) in faces:
				center = (x + w//2, y + h//2)
				frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
				if detect_eyes:
					faceROI = frame_gray[y:y+h,x:x+w]
					#-- In each face, detect eyes
					eyes = self.eyes_cascade.detectMultiScale(faceROI)
					for (x2,y2,w2,h2) in eyes:
						eye_center = (x + x2 + w2//2, y + y2 + h2//2)
						radius = int(round((w2 + h2)*0.25))
						frame = cv2.circle(frame, eye_center, radius, (255, 0, 0 ), 4)

			disp_fps(frame,start_time)
			cv2.imshow('Capture - Face Detection', frame)
		return faces


# Parent Detection class
class Detection:
	def __init__(self,method="cascade",model_path=None,weights_path=None,classes_path=None,conf=0.75):
		if method == "cascade":
			self.detector = CascadeDetector()
		elif method == "yolo":
			self.detector = Dnn(model_path,weights_path,classes_path,conf)
		else:
			logger.error(f"Unknown method specified ({method})")
			logger.opt(colors=True).info("<green>[Solution]:</green> Check Spelling <red>or</red> Check if [<blue>model_path</blue>] and [<blue>model_weight</blue>] are correctly specified.")

		self.method = method # Store selected method as an instance variable of class

	def detect(self,img = None,vid_path = "",display = False):
		
		bboxes = []
		if self.method == "cascade":
			bboxes = self.detector.detectAndDisplay(img,display)
		elif self.method =="yolo":
			if vid_path=="":
				self.detector.detect(img)
			else:
				filename, ext = os.path.splitext(vid_path)
				vid_fomrats = [".mp4", ".avi", ".mov", ".mpeg", ".flv", ".wmv",".webm"]
				if ext in vid_fomrats:
					self.detector.detect_in_video(vid_path,False)
				else:
					logger.error(f"Unknown vid path ({vid_path})")
					exit(0)
		return bboxes



def main():

	print_h("[main]: Investigating OpenCV Detection module.")

	Friends = "Data/NonFree\Friends\Friends_AllClothes.mp4"
	Megamind = "Data/NonFree/Megamind.avi"
	#-- 1. Read the video stream
	cap = cv2.VideoCapture(Megamind)
	if not cap.isOpened:
		logger.error('--(!)Error opening video capture')
		exit(0)
	
	print_h("[a]: Performing (face) Detection using cascade obj detector.")
	detection = Detection()
	while True:
		ret, frame = cap.read()
		if frame is None:
			logger.warning('--(!) No more frames... -- Break!')
			break
		bboxes = detection.detect(frame,display=True)
		if cv2.waitKey(1) == 27: # break on Esc
			break
	cv2.destroyAllWindows()

	
	print_h("[b]: Performing (obj) Detection using yolo-v3 [Dnn-module].\n")
	
	detection_yolo = Detection("yolo")
	test_yolo_in_video = True

	if test_yolo_in_video:		
		# Object detection in video
		dashcam_street = "Data/NonFree\dash_view.webm"
		detection_yolo.detect(vid_path = dashcam_street)
	else:
		# Object detection in Image
		img = cv2.imread('Data\CV\drag_race.jpg')
		detection_yolo.detect(img)

	# Testing yolov7 
	print_h("[c]: Testing yolov7 .\n")
	
	#model_path = r"Data\NonFree\dnn\yolo_v7\yolov7-tiny.cfg"
	#weights_path = r"Data\NonFree\dnn\yolo_v7\yolov7-tiny.weights"

	model_path = r"Data\NonFree\dnn\yolo_v7\detect_mask\yolov7-tiny_TL.cfg"
	weights_path = r"Data\NonFree\dnn\yolo_v7\detect_mask\yolov7-tiny_TL_best.weights"
	classes_path = r"Data\NonFree\dnn\yolo_v7\detect_mask\mask.names"

	#model_path = r"Data\NonFree\dnn\yolo_v7\yolov7.cfg"
	#weights_path = r"Data\NonFree\dnn\yolo_v7\yolov7.weights"


	detection_yolo = Detection("yolo",model_path,weights_path,classes_path,conf = 0.45)
	test_yolo_in_video = True

	if test_yolo_in_video:		
		# Object detection in video
		#dashcam_street = "Data/NonFree\dash_view.webm"
		dashcam_street = "Data/NonFree\School.mp4"
		detection_yolo.detect(vid_path = dashcam_street)
	else:
		# Object detection in Image
		img = cv2.imread('Data\CV\drag_race.jpg')
		detection_yolo.detect(img)
		


if __name__=="__main__":
	main()


	assignment()