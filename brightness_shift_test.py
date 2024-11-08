import cv2
import numpy as np
from collections import deque

class AttackDetector():
    '''
    Detects attacks on vision-based object detection and stores indexes of frames which were attacked.
    '''
    def __init__(self, brightness_threshold=50, frame_interval=20, alpha=0.8, window_size=5):
        self.attacks = {'motion': [],
                        'occlusion': [],
                        'spoofing': [],
                        'tampering': []}
        self.prev_centroids = None
        self.prev_frame = None
        self.brightness_threshold = brightness_threshold
        self.frame_interval = frame_interval
        self.last_stored_frame = None
        self.last_stored_index = -frame_interval
        self.NO_DATA = 0

        # Initialize variables for brightness shift detection
        self.alpha = alpha
        self.prev_filtered_brightness = None
        self.brightness_window = deque(maxlen=window_size)  # Moving window for the last 5 frames

    def detect_attack(self, frame_index, frame, detections, centroids):
        '''
        Defines an attack. An attack is when a frame is rendered useless or misleading by external forces.
        '''
        self.prev_centroids = centroids
        attack = AttackType.NONE

        # Check for occlusion of camera attacks
        occlusion_score = self._detect_occlusion(frame, detections)
        if occlusion_score:
            self.attacks['occlusion'].append(frame_index)
            self.alert_human()
            attack = AttackType.OCCLUSION

        elif self._detect_spoofing(detections):
            self.attacks['spoofing'].append(frame_index)
            self.alert_human()
            attack = AttackType.SPOOFING

        elif frame_index - self.last_stored_index >= self.frame_interval:
            if self.last_stored_frame is not None and self._detect_tampering(frame, self.last_stored_frame):
                self.attacks['tampering'].append(frame_index)
                self.alert_human()
                attack = AttackType.TAMPERING

        # Detect brightness shift
        if self._detect_brightness_shift(frame):
            print("Brightness shift detected at frame:", frame_index)

        self.prev_frame = frame

        return attack

    def _detect_brightness_shift(self, frame):
        """
        Detect substantial brightness shift in a sequence of image frames using an IIR filter and moving window.

        Args:
        - frame: Current image frame (numpy array)

        Returns:
        - Boolean indicating if a substantial brightness shift is detected.
        """
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate the average brightness of the current frame
        avg_brightness = np.mean(gray)

        # Add brightness to the moving window
        self.brightness_window.append(avg_brightness)

        # If the window is not full, return False (not enough data yet)
        if len(self.brightness_window) < self.brightness_window.maxlen:
            return False

        # Apply IIR filter to smooth the brightness values
        if self.prev_filtered_brightness is None:
            filtered_brightness = avg_brightness  # Initialize with the first frame's brightness
        else:
            filtered_brightness = self.alpha * self.prev_filtered_brightness + (1 - self.alpha) * avg_brightness

        # Detect a substantial shift if the difference exceeds the threshold
        if self.prev_filtered_brightness is not None and abs(filtered_brightness - self.prev_filtered_brightness) > self.brightness_threshold:
            self.prev_filtered_brightness = filtered_brightness  # Update previous brightness
            return True  # Shift detected

        # Update the previous filtered brightness value
        self.prev_filtered_brightness = filtered_brightness

        return False
