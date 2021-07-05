#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from vision_msgs.msg import Classification2D, ObjectHypothesis
from sensor_msgs.msg import Image, Range
from cv_bridge import CvBridge
import cv2
import numpy as np

msg = ""
correct = "green"
wrong = "red"
plant = 0
val_min = 0
val_max = float('inf')

def callback(output):
         global msg, plant;
	 msg = output.data[-2:]
         print(msg)
	 if (msg == "12"):
		plant = 0	
         elif (msg == "13"):
		plant = 1 
		
def classification_callback(output):
	hypothesis = output.results[0]
        if plant == hypothesis.id:
		
		rospy.loginfo(correct)
		pub.publish(correct)
	else:
		rospy.loginfo(wrong)
		pub.publish(wrong)

def range_callback(output):

	print(output.field_of_view)
	if 2.5 > output.field_of_view > 0.30:
		val_min = 2000
		val_max = 80000	
		grab_frame(val_min,val_max)
	elif(output.field_of_view > 2.5):
		print("objects are too far")
	else: 
		val_min = 200
		val_max = 30000	
		grab_frame(val_min,val_max)
		
def grab_frame(val_min,val_max):
	cap = cv2.VideoCapture(0)
	while(True):

		ret, image = cap.read()
		img_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
		#cv2.imshow('frame1',image)
		# lower mask
		lower_red = np.array([155,25,0])
		# Upper mask
		upper_red = np.array([179,255,255])
		mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
		nzCount = cv2.countNonZero(mask0)
		print(nzCount)
		#cv2.imshow('frame2',mask0)
		print(val_min)
		print(val_max)
		if (val_max>nzCount>val_min):
			rgb_points_img = CvBridge().cv2_to_imgmsg(np.array(image)[..., ::-1], "rgb8")
			pub1.publish(rgb_points_img)
			break		
		else:
			print("Non Plant")
			pub.publish(wrong)
			break		  				
	cap.release()
	cv2.destroyAllWindows()	

if __name__ == '__main__':

    global msg, correct, wrong, plant, val_min, val_max;
    rospy.init_node('geranium_azalea', anonymous=True)

    rospy.Subscriber('Classification/Control1', String, callback)
    rospy.Subscriber('/Range', Range,range_callback)
    pub1 = rospy.Publisher('/video_source/raw', Image, queue_size=10)
    rospy.Subscriber('/imagenet/classification', Classification2D, classification_callback)
    pub = rospy.Publisher('Prediction', String, queue_size=10)
    
    rospy.spin()
   
