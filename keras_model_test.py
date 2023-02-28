import cv2
import tensorflow.keras as keras
import numpy as np

# Load the saved model
model = keras.models.load_model('cricket_shot_model.h5')

# Set up the video capture
cap = cv2.VideoCapture(0)  # 0 for the default camera

# Define a dictionary to map the class indices to the labels
class_labels = {0: 'drive', 1: 'Legglance Flick', 2: 'pull_shot', 3: 'sweep'}

# Process the video frames and classify the shots
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process the frame to make it compatible with the model input size and format
    processed_frame = cv2.resize(frame, (224, 224))
    processed_frame = np.expand_dims(processed_frame, axis=0)
    processed_frame = processed_frame / 255.0  # normalize the pixel values
    
    # Classify the shot using the loaded model
    pred_probs = model.predict(processed_frame)[0]
    pred_class = class_labels[np.argmax(pred_probs)]
    pred_prob = np.max(pred_probs)
    
    # Display the result
    cv2.putText(frame, f'{pred_class} ({pred_prob:.2f})', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
    cv2.imshow('frame', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
