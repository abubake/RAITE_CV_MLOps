
import numpy as np
import cv2
import playsound
import threading

class attackDetector():
    '''
    Detects attacks on vision based object detection, and stores indexes of frames which were attacked.
    '''
    def __init__(self, brightness_threshold=50, frame_interval=20, warmup_frames=1800):

        self.attacks = {'motion': [],
                        'occlusion': [],
                        'spoofing': [],
                        'tampering': []}  # (then need to be able to assess fp and fn/ F1 score, and accuracy)
        self.prev_centroids = None
        self.prev_frame = None
        self.brightness_threshold = brightness_threshold
        self.frame_interval = frame_interval # frames between which to detect a large change in brightness
        self.warmup_frames = warmup_frames # frames to wait before detecting freeze frame. 10fps * 180s = 1800
        self.last_stored_frame = None
        self.last_stored_index = -frame_interval  # Initialize with -interval to ensure the first frame gets stored

    def alert_human(self):
        '''
        Alerts human operators when an attack is detected
        '''
        #print("I don't want to talk to you no more, you empty-headed animal food trough wiper! I fart in your general direction! Your mother was a hamster, and your father smelt of elderberries!")
        print("Two kinds of people are staying on this beach! The dead and those who are going to die! Now, let’s get the hell out of here!")
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
        # Check for motion attacks (e.g., frame freeze)
        if self.prev_centroids and frame_index > self.warmup_frames:
            for prev, curr in zip(self.prev_centroids, centroids):
                distance = np.linalg.norm(np.array(prev) - np.array(curr))
                if distance < 0.2:
                    self.attacks['motion'].append(frame_index)
                    self.alert_human()
                    break
        
        self.prev_centroids = centroids

        # Check for occlusion of camera attacks
        occlusion_score = self._detect_occlusion(frame, detections)
        if occlusion_score: # when occulsion score, bc it is 0 or 1
            self.attacks['occlusion'].append(frame_index)
            self.alert_human()

        # Spoofing detection
        if self._detect_spoofing(detections):
            self.attacks['spoofing'].append(frame_index)
            self.alert_human()

         # Tampering detection (check against previous frame)
        if frame_index - self.last_stored_index >= self.frame_interval:
            if self.last_stored_frame is not None and self._detect_tampering(frame, self.last_stored_frame):
                self.attacks['tampering'].append(frame_index)
                self.alert_human()

        self.prev_frame = frame


    def _detect_occlusion(self, frame, detections):
        """
        Detect occlusion based on frame brightness or other characteristics.
        
        Args:
        - frame: Current frame from the camera.
        
        Returns:
        - occlusion_score: Proportion of the frame that is considered occluded (0 to 1).
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray_frame)

        if brightness < 50 and len(detections) == 0:  # Adjust this threshold based on the environment
            return 1.0  # Fully occluded
        else:
            return 0.0  # Not occluded
        
    
    def _detect_spoofing(self, detections):
        """
        Detect spoofing based on sudden appearance of objects.
        Returns True if spoofing is detected, otherwise False.
        """
        return len(detections) > 8
    

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
    






    