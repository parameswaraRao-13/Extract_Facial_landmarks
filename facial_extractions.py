import dlib
import cv2
import numpy as np

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)
while True:
    # Getting out image by webcam
    _, image = cap.read()
    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get faces into webcam's image
    rects = detector(gray, 0)
    # rectangles[[(244, 194)(394, 344)]]

    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, rect) #returns object containing 68- x,y coods

        shape = shape_to_np(shape)# convert that into numpy array
        # Draw on our image, all the finded cordinate points (x,y)
        c=0
        for (x, y) in shape:
            c=c+1
            if (c>=1 and c<=17): #the jaw is accessed via points [1, 17].
                cv2.circle(image, (x, y), 2, (255,0,0), -1)
            elif (c>=18 and c<=22): #Right eyebrow is accessed through points [18, 22].
                cv2.circle(image, (x, y), 2, (0,255,0), -1)
            elif (c>=23 and c<=27): #left eyebrow is accessed through points [23, 27].
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            elif (c>=28 and c<=35): #nose is accessed using points [28, 35].
                cv2.circle(image, (x, y), 2, (0, 0, 0), -1)
            elif (c>=37 and c<=42): #right eye is accessed using points [37, 42].
                cv2.circle(image, (x, y), 2, (127,0,255), -1)
            elif (c>=43 and c<=48): #left eye is accessed with points [43, 48].
                cv2.circle(image, (x, y), 2, (127,0,255), -1)
            elif (c>=49 and c<=60): #lips are accessed with points [49,60].
                cv2.circle(image, (x, y), 2, (255,255,0), -1)
            elif (c>=61 and c<=68): #In-mouth is accessed through points [61, 68].
                cv2.circle(image, (x, y), 2, (0,0,255), -1)


    # Show the image
    cv2.imshow("Output", image)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break