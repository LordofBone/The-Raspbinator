#some code/annotations from https://github.com/adafruit/Adafruit_Python_PCA9685/blob/master/examples/simpletest.py 
#and https://thecodacus.com/face-recognition-opencv-train-recognizer/

# import packages
from __future__ import division
import sys
import os
sys.path.append('/usr/local/lib/python2.7/site-packages')
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
from PIL import Image
import time
import Chatbot as b9

# Import the PCA9685 module.
import Adafruit_PCA9685

# Initialise the PCA9685 using the default address (0x40).
pwm = Adafruit_PCA9685.PCA9685()

#set the max/min servo positions
servo_min_v = 420
servo_max_v = 390
servo_min_h = 355
servo_max_h = 400

#set the scaleup/scaledown dividers (used later)
scaledown = 0.2
scaleup = 1/scaledown

# Create the haar cascade
cascadePath = "haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
recognizer = cv2.face.createLBPHFaceRecognizer()

# set font
font = cv2.FONT_HERSHEY_PLAIN

def get_images_and_labels(path):
	# Append all the absolute image paths in a list image_paths
	# Will not read .sad extensions - will only use to test accuracy
	image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad')]
	# images will contain faces
	images = []
	# labels will contain the label assigned to images
	labels = []
	for image_path in image_paths:
		# Read image and convert to grayscale
		image_pil = Image.open(image_path).convert('L')
		
		# Convert the image into numpy array
		image = np.array(image_pil, 'uint8')
		
		# Get the label
		nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
		# Detect the face in image
		faces = faceCascade.detectMultiScale(image)
		# If face detected, append to images and label to labels
		for (x, y, w, h) in faces:
			images.append(image[y: y + h, x: x + w])
			labels.append(nbr)
			#cv2.imshow("Adding faces to training set", image[y: y + h, x: x + w])
			#cv2.waitKey(1000)
			
	# return the images list and labels list
	return images, labels

# Helper function to make setting a servo pulse width simpler.
def set_servo_pulse(channel, pulse):
    pulse_length = 1000000    # 1,000,000 us per second
    pulse_length //= 60       # 60 Hz
    print('{0}us per period'.format(pulse_length))
    pulse_length //= 4096     # 12 bits of resolution
    print('{0}us per bit'.format(pulse_length))
    pulse *= 1000
    pulse //= pulse_length
    pwm.set_pwm(channel, 0, pulse)
    
def face_detect():

	for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
		# grab the raw NumPy array representing the image, then initialize timestamp and occupied/unoccupied text
		image = frame.array
		
		#switch the image to grayscale for quicker processing	
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		
		#scale the image down for further quicker processing (but less accurate)		
		small_img = cv2.resize(gray, (0,0), fx=scaledown, fy=scaledown, interpolation=cv2.INTER_LINEAR)
		
		#equalize the histogram of the grayscale image
		cv2.equalizeHist(small_img, small_img)
		
		# Run the cascade face detection
		faces = faceCascade.detectMultiScale(small_img)	
			
		if len(faces) > 0:		
			for (x, y, w, h) in faces:
				#if faces are detected from the face detection above run the prediction function
				faceid, conf = recognizer.predict(small_img[y: y + h, x: x + w])
				#print the faceid int as well as the confidence level
				print faceid, conf
				#if the confidence level is over 200 identify whos face it is from the trained images
				if conf > 180:
					if faceid == int(1):
						humanid = 'John Connor'
					if faceid == int(2):
						humanid = 'Sarah Connor'
					if faceid == int(3):
						humanid = 'Mike'
				#if the confidence level is below 200 then return 'identify yourself' for the TTS
				else:
					humanid = 'Identify yourself'
		#if no faces detected delete the raw capture from the camera and break the loop to return to the eye movement loop
		else:
			rawCapture.truncate(0)
			break
		#print the humanid for debug purposes
		print humanid
		#call the conversation class from the chatbot and pass along the humanid
		b9.openConversation(humanid)
		#delete the raw capture from the camera
		rawCapture.truncate(0)
		#wait two seconds
		time.sleep(2)

# Set frequency to 60hz, good for servos.
pwm.set_pwm_freq(60)

# Path to the Yale Dataset
path = 'testfaces'
# The folder is the same path as the script
# Call get_images_and_labels function and get the images and the
# corresponding labels
images, labels = get_images_and_labels(path)

# Perform the training
recognizer.train(images, np.array(labels))

# initialize camera and grab a reference raw image
camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(320, 240))

# allow camera to warmup
time.sleep(0.1)

#this loop moves the camera in a square motion to look around, running a face detection before each movement
while True:
    # Move servo on channel O between extremes.
    face_detect()
    pwm.set_pwm(0, 0, servo_min_v)
    time.sleep(2)
    face_detect()
    pwm.set_pwm(0, 0, servo_max_h)
    time.sleep(2)
    face_detect()
    pwm.set_pwm(1, 0, servo_max_v)
    time.sleep(2)
    face_detect()
    pwm.set_pwm(1, 0, servo_min_h)
    face_detect()
    time.sleep(2)
    
# Close windows
cv2.destroyAllWindows()
