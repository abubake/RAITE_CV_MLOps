
from enum import IntEnum
import numpy as np
import cv2
from collections import deque

class AttackType(IntEnum):
    NONE=0
    OCCLUSION=1
    SPOOFING=2
    TAMPERING=3
    MOTION=4
    BRIGHTNESS=5


class AttackDetector():
    '''
    Detects attacks on vision based object detection, and stores indexes of frames which were attacked.
    '''
    def __init__(self, brightness_threshold=50, frame_interval=20, window_size=5):

        self.attacks = {'motion': [],
                        'occlusion': [],
                        'spoofing': [],
                        'tampering': []}  # (then need to be able to assess fp and fn/ F1 score, and accuracy)
        self.prev_centroids = None
        self.prev_frame = None
        self.brightness_threshold = brightness_threshold
        self.frame_interval = frame_interval
        self.last_stored_frame = None
        self.last_stored_index = -frame_interval  # Initialize with -interval to ensure the first frame gets stored
        self.NO_DATA=0

        # Params for the brightness change
        self.brightness_window = deque(maxlen=window_size)
        self.prev_filtered_brightness = None
        self.alpha = 0.5


    def alert_human(self):
        '''
        Alerts human operators when an attack is detected
        '''
        #print("I don't want to talk to you no more, you empty-headed animal food trough wiper! I fart in your general direction! Your mother was a hamster, and your father smelt of elderberries!")
        #print("Two kinds of people are staying on this beach! The dead and those who are going to die! Now, let’s get the hell out of here!")
        # sound_thread = threading.Thread(target=playsound.playsound,args=('/home/basestation/alert.wav',))
        # sound_thread.start()
        # playsound.playsound('/home/basestation/alert.wav')

    def detect_attack(self, frame_index, frame, detections, centroids):
        '''
        Defines an attack. An attack is when a frame is rendered useless or misleading by external forces.

        We track the quantity of attacked frames, and store occurences to a json.

        Args:
        - frame: Current frame from the camera.
        - detections: List of detected bounding boxes from the object detection model.
        - tracker_centroids: List of centroids from the tracking algorithm.
        
        '''
        
        self.prev_centroids = centroids
        attack = AttackType.NONE

        # Check for occlusion of camera attacks
        if self._detect_occlusion(frame,detections): # when occulsion score, bc it is 0 or 1
            self.attacks['occlusion'].append(frame_index)
            self.alert_human()
            attack = AttackType.OCCLUSION

        # elif self._detect_brightness_shift(frame):
        #     self.alert_human()
        #     attack = AttackType.BRIGHTNESS

        # elif self._detect_spoofing(detections):
        #     self.attacks['spoofing'].append(frame_index)
        #     self.alert_human()
        #     attack = AttackType.SPOOFING

        # elif frame_index - self.last_stored_index >= self.frame_interval:
        #     if self.last_stored_frame is not None and self._detect_tampering(frame, self.last_stored_frame):
        #         self.attacks['tampering'].append(frame_index)
        #         self.alert_human()
        #         attack = AttackType.TAMPERING

        self.prev_frame = frame

        return attack


    def _detect_occlusion(self, frame, detections):
        """
        Detect occlusion based on frame brightness or other characteristics.
        
        Args:
        - frame: Current frame from the camera.
        
        Returns:
        - occlusion_score: Proportion of the frame that is considered occluded (0 to 1).
        """
        # 107 is normal for sunny day
        # 17 for blue only image

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray_frame)

        if brightness < 40 or (brightness < 10 and len(detections) == 0):  # Adjust this threshold based on the environment
            return 1.0  # Fully occluded
        else:
            return 0.0  # Not occluded
        

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
            return 5  # Shift detected

        # Update the previous filtered brightness value
        self.prev_filtered_brightness = filtered_brightness

        return False
   
    
    def _detect_spoofing(self, detections):
        """
        Detect spoofing based on sudden appearance of objects.
        Returns True if spoofing is detected, otherwise False.
        """
        return len(detections) > 7
    

    def _detect_tampering(self, current_frame, prev_frame):
        """
        Detect tampering by comparing brightness or color changes between frames.
        Compares every 20 frames.
        """
        current_brightness = np.mean(cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY))
        prev_brightness = np.mean(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY))

        if abs(current_brightness - prev_brightness) > self.brightness_threshold:
            return True  # Significant change detected, indicating possible tampering
        return False
    


# Detect if bounding boxes suddenly change drastically in size for associated ID's
# 

# Smoke
# water bottle
# laser
# screen




    