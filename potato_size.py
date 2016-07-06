# Example USAGE
# python potato_size.py -i potato2small.jpg

# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
#import argparse
import imutils
import cv2

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
rawCapture = PiRGBArray(camera)
# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True, help="path to the input image")
#ap.add_argument("-w", "--width", type=float, required=True, help="width of the left-most object in the image (in inches)")
#args = vars(ap.parse_args())


# allow the camera to warmup
time.sleep(0.1)
 
# grab an image from the camera
camera.capture(rawCapture, format="bgr")
image = rawCapture.array

# load the image, and resize it to 640 by 480 px
#image = cv2.imread(args["image"])

newimage = cv2.resize(image,(640,480))
#cv2.imshow("Image", newimage)
#cv2.waitKey(0)

#image thresholds for lower and upper bounds for potato colors
lower = np.array([0, 100, 100], dtype = "uint8")
upper = np.array([10, 255, 255], dtype = "uint8")

#apply a series of erosions to dilations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
skinMask = cv2.erode(newimage, kernel, iterations = 1)
skinMask = cv2.dilate(skinMask, kernel, iterations = 1)

#blur the image to reduce noise
newimage = cv2.GaussianBlur(skinMask, (3, 3), 0)
mask = cv2.inRange(newimage, lower, upper)
output = cv2.bitwise_and(newimage, newimage, mask = mask)

#cv2.imshow("OUTPUT", output)
#cv2.waitKey(0)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(output,100,200)
#cv2.imshow("EDGES", edged)
#cv2.waitKey(0)

# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cv2.drawContours(newimage, cnts, -1, (0,255,0), 3)
#cv2.imshow("Image", newimage)
#cv2.waitKey(0)

#print out the number of contours found
print(len(cnts))

for c in cnts:
	# ingore small contours
	if cv2.contourArea(c) < 400:
		continue

	# compute the rotated bounding box of the contour
	orig = newimage.copy()
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")

	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
	box = perspective.order_points(box)
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

	# loop over the original points and draw them
	for (x, y) in box:
		cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

	# unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates
	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)

	# compute the midpoint between the top-left and top-right points,
	# followed by the midpoint between the top-righ and bottom-right
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)

	# draw the midpoints on the image
	cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

	# draw lines between the midpoints
	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2)

	# compute the Euclidean distance between the midpoints
	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

	# if the pixels per metric has not been initialized, then
	# compute it as the ratio of pixels to supplied metric
	# (in this case, inches)
	#if pixelsPerMetric is None:
	#	pixelsPerMetric = dB / args["width"]

	# compute the size of the object
	dimA = dA #dA / pixelsPerMetric
	dimB = dB #dB / pixelsPerMetric

	# draw the object sizes on the image
	cv2.putText(orig, "{:}".format(dimA),
		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
	cv2.putText(orig, "{:}".format(dimB),
		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)

	# show the output image
	#cv2.imshow("Image", orig)
	#cv2.waitKey(0)
