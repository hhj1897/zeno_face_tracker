import dlib
from ibug_face_tracker import *


class ZenoFaceTracker(FaceTracker):
    def __init__(self, ert_model_path="", auxiliary_model_path="", face_detector_model_path="",
                 facial_landmark_localiser=None, auxiliary_utility=None, face_detector=None):
        if len(ert_model_path) > 0:
            facial_landmark_localiser = FacialLandmarkLocaliser(ert_model_path)
        if len(auxiliary_model_path) > 0:
            auxiliary_utility = AuxiliaryUtility(auxiliary_model_path)
        super(ZenoFaceTracker, self).__init__(facial_landmark_localiser, auxiliary_utility)
        if len(face_detector_model_path) > 0:
            self._face_detector = dlib.fhog_object_detector(face_detector_model_path)
        elif face_detector is not None:
            self._face_detector = face_detector
        else:
            self._face_detector = dlib.get_frontal_face_detector()
        self._face_detection_countdown = 0xFFFFFFFF

    @property
    def face_detector(self):
        return self._face_detector

    @face_detector.setter
    def face_detector(self, value):
        self._face_detector = value

    def reset(self, reset_face_detection_countdown=True):
        super(ZenoFaceTracker, self).reset(reset_face_detection_countdown)
        if reset_face_detection_countdown:
            self.reset_face_detection_countdown()

    def reset_face_detection_countdown(self):
        self._face_detection_countdown = 0xFFFFFFFF

    def track(self, frame, redetect_face=False, use_bgr_colour_model=True):
        # Convert the frame to grayscale
        if frame.ndim == 3 and frame.shape[2] == 3:
            if use_bgr_colour_model:
                grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            grayscale_frame = frame

        # Are we forced to re-detect the face?
        if redetect_face:
            self.reset()

        if self.has_facial_landmarks:
            retur n super(ZenoFaceTracker, self).track(grayscale_frame, use_bgr_colour_model=use_bgr_colour_model)
        elif
