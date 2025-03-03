import cv2
import mediapipe as mp
import pyautogui
import util  # Ensure this is the correct path to your util module
import psutil
import time
from pynput.mouse import Button, Controller

mouse = Controller()

screen_width, screen_height = pyautogui.size()

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

cv2.setUseOptimized(True)
cv2.setNumThreads(4)

# Helper function to calculate the Euclidean distance
def calculate_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        index_finger_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
        return index_finger_tip
    return None

def move_mouse(index_finger_tip):
    if index_finger_tip is not None:
        x = int(index_finger_tip.x * screen_width)
        y = int(index_finger_tip.y * screen_height)
        pyautogui.moveTo(x, y)

def detect_gesture(frame, landmark_list, processed):
    if len(landmark_list) >= 21:
        # Get coordinates of important landmarks
        index_tip = landmark_list[mpHands.HandLandmark.INDEX_FINGER_TIP]
        index_base = landmark_list[mpHands.HandLandmark.INDEX_FINGER_PIP]
        pinky_tip = landmark_list[mpHands.HandLandmark.PINKY_TIP]
        pinky_base = landmark_list[mpHands.HandLandmark.PINKY_PIP]

        # Measure distances
        index_distance = calculate_distance(index_tip, index_base)
        pinky_distance = calculate_distance(pinky_tip, pinky_base)

        # Define threshold (adjust as needed)
        threshold = 0.04  # Adjust based on testing and screen size

        # Check for gestures
        if pinky_distance < threshold:
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif index_distance < threshold:
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Mouse movement
        index_finger_tip = find_finger_tip(processed)
        if index_finger_tip is not None:
            move_mouse(index_finger_tip)

def process_frame(frame, landmark_list, processed):
    detect_gesture(frame, landmark_list, processed)

def main():
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    skip_frames = 2

    processing_times = []
    memory_usages = []
    cpu_usages = []

    try:
        while cap.isOpened():
            start_time = time.time()  # Start timer

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            landmark_list = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x, lm.y))

            process_frame(frame, landmark_list, processed)

            # Measure performance metrics
            processing_time = time.time() - start_time
            processing_times.append(processing_time)

            memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # Convert to MB
            cpu_usage = psutil.cpu_percent()
            memory_usages.append(memory_usage)
            cpu_usages.append(cpu_usage)

            cv2.putText(frame, f'Time: {processing_time:.2f}s', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f'Mem: {memory_usage:.2f}MB', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f'CPU: {cpu_usage:.2f}%', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()