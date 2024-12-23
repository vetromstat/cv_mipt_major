import cv2
import numpy as np

def find_road_number(image: np.ndarray) -> int:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    thresholds = {
        'green': (np.array([40, 100, 100]), np.array([80, 255, 255])),
        'red1': (np.array([0, 100, 100]), np.array([10, 255, 255])),
        'red2': (np.array([170, 100, 100]), np.array([180, 255, 255])),
    }

    masks = {
        'green': cv2.inRange(hsv, thresholds['green'][0], thresholds['green'][1]),
        'red': cv2.inRange(hsv, thresholds['red1'][0], thresholds['red1'][1]) | 
               cv2.inRange(hsv, thresholds['red2'][0], thresholds['red2'][1]),
    }

    contours_green, _ = cv2.findContours(masks['green'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_red, _ = cv2.findContours(masks['red'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width, _ = image.shape
    lane_count = 5
    lane_width = width // lane_count
    lane_status = [False] * lane_count

    for cnt in contours_red:
        x, _, _, _ = cv2.boundingRect(cnt)
        lane_idx = x // lane_width
        if lane_idx < lane_count:
            lane_status[lane_idx] = True

    for idx, occupied in enumerate(lane_status):
        if not occupied:
            return idx

    return -1