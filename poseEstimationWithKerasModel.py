import cv2
import keras
import numpy as np
import mediapipe as mp



# Load the saved model
model = keras.models.load_model('cricket_shot_model.h5')

mp_draw = mp.solutions.drawing_utils # use to draw skeleton on body
mp_pose = mp.solutions.pose # human pose 
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) # constructor

def pose_classification():
    camera_video = cv2.VideoCapture(0)
    camera_video.set(3,1280)
    camera_video.set(4,960)

    cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

    # Define a dictionary to map the class indices to the labels
    class_labels = {0: 'drive', 1: 'Legglance Flick', 2: 'pull_shot', 3: 'sweep'}

    # Process the video frames and classify the shots
    while True:
        ret, frame = camera_video.read()
        if not ret:
            break
        
        # Process the frame to make it compatible with the model input size and format
        processed_frame = cv2.resize(frame, (224, 224))
        processed_frame = np.expand_dims(processed_frame, axis=0)
        processed_frame = processed_frame / 255.0  # normalize the pixel values


        # convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # process the frame for pose detection
        pose_results = pose.process(frame_rgb)
        # print(pose_results.pose_landmarks)
        
        # Draw landmarks and connections with custom colors
        drawing_spec = mp_draw.DrawingSpec(color=(235, 206, 135), thickness=2, circle_radius=2)
        connection_spec = mp_draw.DrawingSpec(color=(235, 206, 135), thickness=2, circle_radius=2)

        # draw skeleton on the frame
        mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec = drawing_spec, connection_drawing_spec = connection_spec)


        # Classify the shot using the loaded model
        pred_probs = model.predict(processed_frame)[0]
        pred_class = class_labels[np.argmax(pred_probs)]
        pred_prob = np.max(pred_probs)

        if pred_prob < 0.99:
            pred_class = "Normal Stand"
    
        # Display the result
        cv2.putText(frame, f'{pred_class} ({pred_prob:.2f})', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
        cv2.imshow('frame', frame)
    
        # Press 'q' to exit
        if cv2.waitKey(1) == ord('q'):
            break

    camera_video.release()
    cv2.destroyAllWindows()

