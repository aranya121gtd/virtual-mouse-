import numpy as np

def get_angle(a, b, c):
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    if angle > 180:
        angle = 360 - angle
    return angle

def get_distance(landmark_list):
    if len(landmark_list) < 2:
        raise ValueError("landmark_list must contain at least two points")
    (x1, y1), (x2, y2) = landmark_list[0], landmark_list[1]
    distance = np.hypot(x2 - x1, y2 - y1) 
    return distance