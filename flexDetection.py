import cv2
import mediapipe as mp
import numpy as np
import pygame

def calculate_angle(points):
    a, b, c = points
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def main():
    cap = cv2.VideoCapture(0)  # Use the default camera (webcam)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.9, min_tracking_confidence=0.9)

    # Initialize pygame mixer for sound
    pygame.mixer.init()

    prev_angle = None
    flexion_triggered = False
    sound_triggered = False

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the BGR frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame using MediaPipe
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Get the right arm landmarks
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

            # Draw the right arm lines
            if all([right_shoulder, right_elbow, right_wrist]):
                right_shoulder_x, right_shoulder_y = int(right_shoulder.x * frame.shape[1]), int(right_shoulder.y * frame.shape[0])
                right_elbow_x, right_elbow_y = int(right_elbow.x * frame.shape[1]), int(right_elbow.y * frame.shape[0])
                right_wrist_x, right_wrist_y = int(right_wrist.x * frame.shape[1]), int(right_wrist.y * frame.shape[0])

                cv2.line(frame, (right_shoulder_x, right_shoulder_y), (right_elbow_x, right_elbow_y), (255, 0, 0), 3)
                cv2.line(frame, (right_elbow_x, right_elbow_y), (right_wrist_x, right_wrist_y), (255, 0, 0), 3)

                # Calculate the angle between the right shoulder, elbow, and wrist points
                angle = calculate_angle(np.array([(right_shoulder_x, right_shoulder_y),
                                                  (right_elbow_x, right_elbow_y),
                                                  (right_wrist_x, right_wrist_y)]))

                # Display the angle on the frame
                cv2.putText(frame, f"Angle: {angle:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Check if the angle is approximately 90 degrees and it's decreasing (flexion)
                if prev_angle is not None and angle < 90 and angle < prev_angle:
                    flexion_triggered = True
                else:
                    flexion_triggered = False

                if flexion_triggered and not sound_triggered:
                    # Load and play the sound effect
                    pygame.mixer.music.load("vine-boom.mp3")  # Replace with the actual path to your sound file
                    pygame.mixer.music.play()
                    sound_triggered = True
                    # Add a text overlay when the sound effect is triggered
                    cv2.putText(frame, "Sound Effect Triggered!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif angle >= 90:
                    sound_triggered = False

                prev_angle = angle

        cv2.imshow("Right Arm Angle Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
