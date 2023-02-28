import cv2
import math
import mediapipe as mp
import pandas as pd
#import pyodbc
import numpy as np
import matplotlib.pyplot as plt
from time import time
import sklearn
import pickle
loaded_model = pickle.load(open(r"Random_forest_model.sav", 'rb'))
# df = pd.DataFrame
df=pd.DataFrame
# initalise pose estimator
mp_draw = mp.solutions.drawing_utils # use to draw skeleton on body
mp_pose = mp.solutions.pose # human pose 
pose = mp_pose.Pose(static_image_mode=False,min_detection_confidence=0.5, min_tracking_confidence=0.5) # constructor

def detectPose(image, pose, display=True):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image, 
                 and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    values=[12,11,14, 13, 23, 24,25,26,27,28]
    i=0
    # Check if any landmarks are detected.
    if results.pose_landmarks:
    
        # Draw Pose landmarks on the output image.
        mp_draw.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        
        # Iterate over the detected landmarks.
        for index in values:
            
            #condition for landmarks
            
            # Append the landmark into the list.
            # landmarks.append((int(results.pose_landmarks.landmark[index].x * width), int(results.pose_landmarks.landmark[index].y * height),
            #                       (results.pose_landmarks.landmark[index].z * width)))
            landmarks.append((int(results.pose_landmarks.landmark[index].x * width)))
            landmarks.append((int(results.pose_landmarks.landmark[index].y * height)))
            landmarks.append((int(results.pose_landmarks.landmark[index].z * width)))
        print(landmarks)
        # df = pd.DataFrame(landmarks, columns =["Right Shoulder (x)", "Right Shoulder (y)", "Right Shoulder (z)", "Left Shoulder (x)", "Left Shoulder (y)", "Left Shoulder (z)", "Right Elbow (x)", "Right Elbow (y)", "Right Elbow (z)", "Left Elbow (x)", "Left Elbow (y)","Left Elbow (z)", "Right Hip (x)", "Right Hip (y)", "Right Hip (z)", "Left Hip (x)", "Left Hip (y)", "Left Hip (z)", "Right Knee (x)", "Right Knee (y)", "Right Knee (z)", "Left Knee (x)", "Left Knee (y)", "Left Knee (z)", "Right Ankle (x)", "Right Ankle (y)", "Right Ankle (z)", "Left Ankle (x)", "Left Ankle (y)", "Left Ankle (z)"])
    # Check if the original input image and the resultant image are specified to be displayed.
    if display:
    
        # Display the original input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
        # Also Plot the Pose landmarks in 3D.
        mp_draw.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
    # Otherwise
    else:
        
        # Return the output image and the found landmarks.
        return output_image, landmarks
def calculateAngle(landmark1, landmark2, landmark3):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.

    '''

    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Check if the angle is less than zero.
    if angle < 0:

        # Add 360 to the found angle.
        angle += 360
    
    # Return the calculated angle.
    return angle

def classifyPose(landmarks, output_image, display=False):
    '''
    This function classifies yoga poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''
    
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    #----------------------------------------------------------------------------------------------------------------
    # df = pd.DataFrame(landmarks, columns =["Right Shoulder (x)", "Right Shoulder (y)", "Right Shoulder (z)", "Left Shoulder (x)", "Left Shoulder (y)", "Left Shoulder (z)", "Right Elbow (x)", "Right Elbow (y)", "Right Elbow (z)", "Left Elbow (x)", "Left Elbow (y)","Left Elbow (z)", "Right Hip (x)", "Right Hip (y)", "Right Hip (z)", "Left Hip (x)", "Left Hip (y)", "Left Hip (z)", "Right Knee (x)", "Right Knee (y)", "Right Knee (z)", "Left Knee (x)", "Left Knee (y)", "Left Knee (z)", "Right Ankle (x)", "Right Ankle (y)", "Right Ankle (z)", "Left Ankle (x)", "Left Ankle (y)", "Left Ankle (z)"])
    df = pd.DataFrame(landmarks).T
    df.columns=["Right Shoulder (x)", "Right Shoulder (y)", "Right Shoulder (z)", "Left Shoulder (x)", "Left Shoulder (y)", "Left Shoulder (z)", "Right Elbow (x)", "Right Elbow (y)", "Right Elbow (z)", "Left Elbow (x)", "Left Elbow (y)","Left Elbow (z)", "Right Hip (x)", "Right Hip (y)", "Right Hip (z)", "Left Hip (x)", "Left Hip (y)", "Left Hip (z)", "Right Knee (x)", "Right Knee (y)", "Right Knee (z)", "Left Knee (x)", "Left Knee (y)", "Left Knee (z)", "Right Ankle (x)", "Right Ankle (y)", "Right Ankle (z)", "Left Ankle (x)", "Left Ankle (y)", "Left Ankle (z)"]
    results=loaded_model.predict(df)
    label = str(results)
                
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)  
    
    # Write the label on the output image. 
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 5)
    
    # Check if the resultant image is specified to be displayed.
    if display:
    
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
    else:
        
        # Return the output image and the classified label.
        return output_image, label

# Setup Pose function for video.
def PoseCalculation():
    pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Initialize the VideoCapture object to read from the webcam.
    camera_video = cv2.VideoCapture(0)
    camera_video.set(3,1280)
    camera_video.set(4,960)

# Initialize a resizable window.
    cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

# Iterate until the webcam is accessed successfully.
    while camera_video.isOpened():
    
    # Read a frame.
        ok, frame = camera_video.read()
    
    # Check if frame is not read properly.
        if not ok:
        
        # Continue to the next iteration to read the next frame and ignore the empty camera frame.
            continue
    
    # Flip the frame horizontally for natural (selfie-view) visualization.
        frame = cv2.flip(frame, 1)
    
    # Get the width and height of the frame
        frame_height, frame_width, _ =  frame.shape
    
    # Resize the frame while keeping the aspect ratio.
        frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
    
    # Perform Pose landmark detection.
        frame, landmarks = detectPose(frame, pose_video, display=False)
    
    # Check if the landmarks are detected.
        if landmarks:
        
        # Perform the Pose Classification.
            frame, _ = classifyPose(landmarks, frame, display=False)
    
    # Display the frame.
        cv2.imshow('Pose Classification', frame)
    
    # Wait until a key is pressed.
    # Retreive the ASCII code of the key pressed
        # cv2.createButton("End Session",EndSession,None,cv2.QT_PUSH_BUTTON,1)
        
        k = cv2.waitKey(1) & 0xFF
    # Check if 'ESC' is pressed.
        if(k == 27):
        
        # Break the loop.
            break

# # Release the VideoCapture object and close the windows.  
    camera_video.release()
    cv2.destroyAllWindows()


# def EndSession(*argv):
#     print("pressed")
#     # argv[0].release()
#     # cv2.destroyAllWindows()
    
# PoseCalculation()