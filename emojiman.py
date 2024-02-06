
import cv2
import mediapipe as mp
import numpy as np
import time








mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)


# Load the emoji image
emoji_img = cv2.imread('emoji2.png', -1)  # Replace 'path_to_emoji_image.png' with the actual path
left_hand_img = cv2.imread('left_hand.png', -1)  # Replace with actual path to left hand image
right_hand_img = cv2.imread('right_hand.png', -1)  # Replace with actual path to right hand image
left_boot_img = cv2.imread('left_boot.png', -1)  # Replace with actual path to left boot image
right_boot_img = cv2.imread('right_boot.png', -1)  # Replace with actual path to right boot image
cigarette_img = cv2.imread('cigarette.png', -1)  # Replace with actual path
smoke_img = cv2.imread('smoke.png', -1)  # Replace with actual path


# State flags and timers
cigarette_close = False
smoke_display_time = 2  # Seconds
smoke_start_time = None


# Function to overlay an image on top of another
def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    x, y = pos
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    for c in range(0, 3):
        img[y1:y2, x1:x2, c] = img[y1:y2, x1:x2, c] * (1.0 - alpha_mask[y1o:y2o, x1o:x2o]) + \
                               img_overlay[y1o:y2o, x1o:x2o, c] * alpha_mask[y1o:y2o, x1o:x2o]


# Function to calculate distance between two points
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Render detections
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=10, circle_radius=2), 
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=10, circle_radius=2)
        )

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            left_index = landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value]
            right_index = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value]

            # Coordinates of the nose
            nose_x = int(nose.x * image.shape[1])
            nose_y = int(nose.y * image.shape[0])

            # Resize emoji and overlay it
            emoji_scaled = cv2.resize(emoji_img, (100, 100))  # Adjust size as needed
            alpha_s = emoji_scaled[:, :, 3] / 255.0
            overlay_image_alpha(image, emoji_scaled[:, :, 0:3], (nose_x - 50, nose_y - 65), alpha_s)

            # Overlay Left Hand
            left_hand_x = int(left_index.x * image.shape[1])
            left_hand_y = int(left_index.y * image.shape[0])
            left_hand_scaled = cv2.resize(left_hand_img, (50, 50))  # Adjust size as needed
            alpha_lh = left_hand_scaled[:, :, 3] / 255.0
            overlay_image_alpha(image, left_hand_scaled[:, :, 0:3], (left_hand_x - 12, left_hand_y - 12), alpha_lh)

            # Overlay Right Hand
            right_hand_x = int(right_index.x * image.shape[1])
            right_hand_y = int(right_index.y * image.shape[0])
            right_hand_scaled = cv2.resize(right_hand_img, (50, 50))  # Adjust size as needed
            alpha_rh = right_hand_scaled[:, :, 3] / 255.0
            overlay_image_alpha(image, right_hand_scaled[:, :, 0:3], (right_hand_x - 12, right_hand_y - 12), alpha_rh)

            # Overlay Left Boot
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            left_ankle_x = int(left_ankle.x * image.shape[1])
            left_ankle_y = int(left_ankle.y * image.shape[0])
            left_boot_scaled = cv2.resize(left_boot_img, (100, 100))  # Adjust size as needed
            alpha_lb = left_boot_scaled[:, :, 3] / 255.0
            overlay_image_alpha(image, left_boot_scaled[:, :, 0:3], (left_ankle_x - 25, left_ankle_y - 50), alpha_lb)

            # Overlay Right Boot
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            right_ankle_x = int(right_ankle.x * image.shape[1])
            right_ankle_y = int(right_ankle.y * image.shape[0])
            right_boot_scaled = cv2.resize(right_boot_img, (100, 100))  # Adjust size as needed
            alpha_rb = right_boot_scaled[:, :, 3] / 255.0
            overlay_image_alpha(image, right_boot_scaled[:, :, 0:3], (right_ankle_x - 70, right_ankle_y - 50), alpha_rb)

            distance = calculate_distance(nose.x * image.shape[1], nose.y * image.shape[0], 
                                          left_index.x * image.shape[1], left_index.y * image.shape[0])

            if distance < 150:
                cigarette_close = True
                # Overlay Cigarette
                cigarette_scaled = cv2.resize(cigarette_img, (60, 20))  # Adjust size as needed
                alpha_cig = cigarette_scaled[:, :, 3] / 255.0
                overlay_image_alpha(image, cigarette_scaled[:, :, 0:3], 
                                    (int(left_index.x * image.shape[1] - 10) + 50, 
                                     int(left_index.y * image.shape[0] - 20) + 20), alpha_cig)

            elif cigarette_close:
                cigarette_close = False
                smoke_start_time = time.time()

            # Display smoke for 2 seconds after cigarette moves away
            if smoke_start_time and (time.time() - smoke_start_time < smoke_display_time):
                smoke_scaled = cv2.resize(smoke_img, (80, 80))  # Adjust size as needed
                alpha_smoke = smoke_scaled[:, :, 3] / 255.0
                overlay_image_alpha(image, smoke_scaled[:, :, 0:3],
                                    (int(nose.x * image.shape[1] - 50) - 100, 
                                     int(nose.y * image.shape[0] - 50) + 20), alpha_smoke)

        except:
            pass


        cv2.imshow('EYE T.V.', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
