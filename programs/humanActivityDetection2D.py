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

print("")
print("")
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

print("")
print("Loading humanActivityDetection2D module ...")
print("")

print("")
print("**************************************************************************")
print("YARP configuration:")
print("**************************************************************************")
print("")
print("Initializing YARP network ...")
print("")

# Init YARP Network
yarp.Network.init()

print("")
print("[INFO] Opening image input port with name /humanActivityDetection2D/img:i ...")
print("")

# Open input image port
humanActivityDetection2D_portIn = yarp.BufferedPortImageRgb()
humanActivityDetection2D_portNameIn = '/humanActivityDetection2D/img:i'
humanActivityDetection2D_portIn.open(humanActivityDetection2D_portNameIn)

print("")
print("[INFO] Opening image output port with name /humanActivityDetection2D/img:o ...")
print("")

# Open output image port
humanActivityDetection2D_portOut = yarp.Port()
humanActivityDetection2D_portNameOut = '/humanActivityDetection2D/img:o'
humanActivityDetection2D_portOut.open(humanActivityDetection2D_portNameOut)

print("")
print("[INFO] Opening data output port with name /humanActivityDetection2D/data:o ...")
print("")

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
print("[INFO] YARP network configured correctly.")
print("")

print("")
print("**************************************************************************")
print("Loading models:")
print("**************************************************************************")
print("")
print("[INFO] Loading models at " + str(datetime.datetime.now()) + " ...")
print("")

# Load labels
print("")
print("[INFO] Loading label classes at " + str(datetime.datetime.now()) + " ...")
print("")

labelClases = open("./../models/humanActivityLabels.txt").read().strip().split("\n")

print("")
print("[INFO] Labels classes loaded correctly.")
print("")

print("")
print("[INFO] Loading DNN model at " + str(datetime.datetime.now()) + " ...")
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

# Control loop
loopControlReadImage = 0

while int(loopControlReadImage) == 0:

    try:
        print("")
        print("**************************************************************************")
        print("Waiting for input image source:")
        print("**************************************************************************")
        print("")
        print("[INFO] Waiting input image source at " + str(datetime.datetime.now()) + " ...")
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

        print("")
        print("**************************************************************************")
        print("Analyzing image source:")
        print("**************************************************************************")
        print("")
        print("[INFO] Analyzing image source at " + str(datetime.datetime.now()) + " ...")
        print("")

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

        # Print processed data
        print("")
        print("**************************************************************************")
        print("Results resume:")
        print("**************************************************************************")
        print("")
        print("[RESULTS] Human activity detection results:")
        print("")
        print("[DETECTION] Detected activity: " + str(detectedActivity))
        print("[DATE] Detection time: "+ str(datetime.datetime.now()))
        print("")

        # Sending processed detection
        outputBottleHumanActivityDetection2D.clear()
        outputBottleHumanActivityDetection2D.addString("DETECTION:")
        outputBottleHumanActivityDetection2D.addString(str(detectedActivity))
        outputBottleHumanActivityDetection2D.addString("DATE:")
        outputBottleHumanActivityDetection2D.addString(str(datetime.datetime.now()))
        humanActivityDetection2D_portOutDet.write(outputBottleHumanActivityDetection2D)

        # Sending processed image
        print("")
        print("[INFO] Sending processed image ...")
        print("")

        out_buf_array[:,:] = in_buf_array
        humanActivityDetection2D_portOut.write(out_buf_image)

    except:
        print("")
        print("[ERROR] Empty frame.")
        print("")

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
