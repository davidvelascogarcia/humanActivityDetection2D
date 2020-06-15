'''
  * ************************************************************
  *      Program: Human Activity Detection 2D Module
  *      Type: Python
  *      Author: David Velasco Garcia @davidvelascogarcia
  * ************************************************************
  *
  * | INPUT PORT                           | CONTENT                                                 |
  * |--------------------------------------|---------------------------------------------------------|
  * | /humanActivityDetection2D/img:i      | Input image                                             |
  *
  *
  * | OUTPUT PORT                          | CONTENT                                                 |
  * |--------------------------------------|---------------------------------------------------------|
  * | /humanActivityDetection2D/img:o      | Output image with activity detection analysis           |
  * | /humanActivityDetection2D/data:o     | Output result, activity analysis data                   |
  *
'''

# Libraries
import argparse
import cv2
import datetime
import imutils
import numpy as np
import sys
import time
import yarp

print("**************************************************************************")
print("**************************************************************************")
print("                   Program: Human Activity Detection 2D                   ")
print("                     Author: David Velasco Garcia                         ")
print("                             @davidvelascogarcia                          ")
print("**************************************************************************")
print("**************************************************************************")

print("")
print("Starting system ...")

print("")
print("Loading humanActivityDetection2D module ...")


print("")
print("")
print("**************************************************************************")
print("YARP configuration:")
print("**************************************************************************")
print("")
print("")
print("Initializing YARP network ...")

# Init YARP Network
yarp.Network.init()


print("")
print("[INFO] Opening image input port with name /humanActivityDetection2D/img:i ...")

# Open input image port
humanActivityDetection2D_portIn = yarp.BufferedPortImageRgb()
humanActivityDetection2D_portNameIn = '/humanActivityDetection2D/img:i'
humanActivityDetection2D_portIn.open(humanActivityDetection2D_portNameIn)

print("")
print("[INFO] Opening image output port with name /humanActivityDetection2D/img:o ...")

# Open output image port
humanActivityDetection2D_portOut = yarp.Port()
humanActivityDetection2D_portNameOut = '/humanActivityDetection2D/img:o'
humanActivityDetection2D_portOut.open(humanActivityDetection2D_portNameOut)

print("")
print("[INFO] Opening data output port with name /humanActivityDetection2D/data:o ...")

# Open output data port
humanActivityDetection2D_portOutDet = yarp.Port()
humanActivityDetection2D_portNameOutDet = '/humanActivityDetection2D/data:o'
humanActivityDetection2D_portOutDet.open(humanActivityDetection2D_portNameOutDet)

# Create data bootle
outputBottleHumanActivityDetection2D = yarp.Bottle()

# Image size
image_w = 640
image_h = 480

# Prepare input image buffer
in_buf_array = np.ones((image_h, image_w, 3), np.uint8)
in_buf_image = yarp.ImageRgb()
in_buf_image.resize(image_w, image_h)
in_buf_image.setExternal(in_buf_array.data, in_buf_array.shape[1], in_buf_array.shape[0])

# Prepare output image buffer
out_buf_image = yarp.ImageRgb()
out_buf_image.resize(image_w, image_h)
out_buf_array = np.zeros((image_h, image_w, 3), np.uint8)
out_buf_image.setExternal(out_buf_array.data, out_buf_array.shape[1], out_buf_array.shape[0])


print("")
print("")
print("**************************************************************************")
print("Loading models:")
print("**************************************************************************")
print("")
print("")
print("Loading models ...")
print("")

# Load labels
print("")
print("Loading label classes ...")
print("")
labelClases = open("./../models/humanActivityLabels.txt").read().strip().split("\n")
print("")
print("[INFO] Labels classes loaded correctly.")
print("")

print("")
print("Loading DNN model ...")
print("")
dnnNet = cv2.dnn.readNet("./../models/humanActivityModel.onnx")
print("")
print("[INFO] Models loaded correctly.")
print("")


print("")
print("Configuring samples ...")
print("")
sampleDuration = 16
sampleSize = 112
print("")
print("[INFO] Samples configured correctly.")
print("")

print("")
print("")
print("**************************************************************************")
print("Waiting for input image source:")
print("**************************************************************************")
print("")
print("")
print("Waiting input image source ...")
print("")

# Control loop
loopControlReadImage = 0

while int(loopControlReadImage) == 0:

    print("")
    print("")
    print("**************************************************************************")
    print("Analyzing image source:")
    print("**************************************************************************")
    print("")
    print("Analyzing image source ...")
    print("")

    # Frames array
    rgbFrames = []

    # Receive frames array with sampleDuration
    for i in range(0, sampleDuration):

        # Receive image source
        frame = humanActivityDetection2D_portIn.read()

        # Buffer processed image
        in_buf_image.copy(frame)
        assert in_buf_array.__array_interface__['data'][0] == in_buf_image.getRawImage().__int__()

        # YARP -> OpenCV
        rgbFrame = in_buf_array[:, :, ::-1]

        # Resize rgbFrame
        rgbFrame = imutils.resize(rgbFrame, 320, 240)
        rgbFrames.append(rgbFrame)

	# Analyzing rgb frames array
    blobFromImagesObject = cv2.dnn.blobFromImages(rgbFrames, 1.0, (sampleSize, sampleSize), (114.7748, 107.7354, 99.4750), swapRB = True, crop = True)
    blobFromImagesObject = np.transpose(blobFromImagesObject, (1, 0, 2, 3))
    blobFromImagesObject = np.expand_dims(blobFromImagesObject, axis = 0)

    # Recognizing human activity
    dnnNet.setInput(blobFromImagesObject)
    predictionResults = dnnNet.forward()

    # Compare with label classes index
    detectedActivity = labelClases[np.argmax(predictionResults)]
    detectedActivity = str(detectedActivity)

    # draw the predicted activity on the frame
    cv2.rectangle(in_buf_array, (0, 0), (300, 40), (0, 0, 0), -1)
    cv2.putText(in_buf_array, detectedActivity, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    print("")
    print("[INFO] Image source analysis done correctly.")
    print("")

    # Get time Detection
    timeDetection = datetime.datetime.now()

    # Print processed data
    print("")
    print("**************************************************************************")
    print("Results resume:")
    print("**************************************************************************")
    print("")
    print("[RESULTS] Human activity detection results:")
    print("Detected activity: " + str(detectedActivity))
    print("[INFO] Detection time: "+ str(timeDetection))

    # Sending processed detection
    outputBottleHumanActivityDetection2D.clear()
    outputBottleHumanActivityDetection2D.addString("Detected activity:")
    outputBottleHumanActivityDetection2D.addString(str(detectedActivity))
    humanActivityDetection2D_portOutDet.write(outputBottleHumanActivityDetection2D)

    # Sending processed image
    print("")
    print("[INFO] Sending processed image ...")
    print("")
    out_buf_array[:,:] = in_buf_array
    humanActivityDetection2D_portOut.write(out_buf_image)

# Close ports
print("[INFO] Closing ports ...")
humanActivityDetection2D_portIn.close()
humanActivityDetection2D_portOut.close()
humanActivityDetection2D_portOutDet.close()

print("")
print("")
print("**************************************************************************")
print("Program finished")
print("**************************************************************************")
print("")
print("humanActivityDetection2D program closed correctly.")
print("")
