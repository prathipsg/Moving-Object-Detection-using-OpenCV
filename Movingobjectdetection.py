import cv2
import imutils

# Attempt to access the default camera (index 0)
cam = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cam.isOpened():
    print("Error: Could not access the camera.")
    exit()

firstFrame = None
area = 500
frameResetCounter = 0  # Counter to reset the first frame periodically

while True:
    # Capture frame from the camera
    ret, img = cam.read()

    # Check if the frame was successfully captured
    if not ret:
        print("Failed to grab frame.")
        break

    # Resize the image if it was captured
    img = imutils.resize(img, width=500)
    
    # Convert the image to grayscale and apply Gaussian blur
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)
    
    # Initialize the first frame, or reset it every 100 frames to adjust for changes
    if firstFrame is None or frameResetCounter >= 100:
        firstFrame = gaussianImg
        frameResetCounter = 0  # Reset the counter once firstFrame is updated
        continue
    
    # Compute the difference between the current frame and the first frame
    imgDiff = cv2.absdiff(firstFrame, gaussianImg)
    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1]
    threshImg = cv2.dilate(threshImg, None, iterations=2)
    
    # Find contours of the thresholded image
    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Variable to track if movement is detected
    movementDetected = False
    
    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Moving Object detected"
        print(text)
        cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        movementDetected = True
    
    # If no movement was detected, show "Normal"
    if not movementDetected:
        text = "Normal"
        print(text)
        cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow("cameraFeed", img)
    
    # Increment the reset counter
    frameResetCounter += 1

    # Break the loop if 'q' is pressed
    key = cv2.waitKey(10)
    if key == ord("q"):
        break

# Release the camera and close windows
cam.release()
cv2.destroyAllWindows()
