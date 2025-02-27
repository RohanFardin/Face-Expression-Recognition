import cv2
import numpy as np
import tensorflow as tf

# Configuration
CONFIG = {
    "model_path": "models/emotion_model_v2.h5",
    "haar_cascade_path": cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
    "emotion_labels": ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
    "target_size": (48, 48),
    "confidence_threshold": 0.65  # Only show predictions with >65% confidence
}

class EmotionDetector:
    def __init__(self):
        # Load model
        self.model = tf.keras.models.load_model(CONFIG["model_path"])
        
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(CONFIG["haar_cascade_path"])
        if self.face_cascade.empty():
            raise SystemExit("Error loading face detection cascade. Check OpenCV installation.")

    def detect_emotions(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Process each face
        for (x, y, w, h) in faces:
            # Extract and preprocess face ROI
            face_roi = gray[y:y+h, x:x+w]
            resized = cv2.resize(face_roi, CONFIG["target_size"], interpolation=cv2.INTER_AREA)
            normalized = resized / 255.0
            input_tensor = np.expand_dims(normalized, axis=(0, -1))
            
            # Make prediction
            predictions = self.model.predict(input_tensor)
            confidence = np.max(predictions)
            label = CONFIG["emotion_labels"][np.argmax(predictions)]
            
            # Only show predictions above confidence threshold
            if confidence > CONFIG["confidence_threshold"]:
                # Draw bounding box and label
                color = (0, 255, 0)  # Green
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                text = f"{label}: {confidence:.0%}"
                cv2.putText(frame, text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        return frame

def main():
    detector = EmotionDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise SystemExit("Cannot open webcam")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = detector.detect_emotions(frame)
            cv2.imshow('Emotion Detection', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()