import cv2
import dlib
from ibug_face_tracker import *


class ZenoFaceTracker(FaceTracker):
    def __init__(self, ert_model_path="", auxiliary_model_path="", face_detection_model_path="",
                 facial_landmark_localiser=None, auxiliary_utility=None, face_detector=None):
        if len(ert_model_path) > 0:
            facial_landmark_localiser = FacialLandmarkLocaliser(ert_model_path)
        if len(auxiliary_model_path) > 0:
            auxiliary_utility = AuxiliaryUtility(auxiliary_model_path)
        super(ZenoFaceTracker, self).__init__(facial_landmark_localiser, auxiliary_utility)
        if len(face_detection_model_path) > 0:
            self._face_detector = dlib.fhog_object_detector(face_detection_model_path)
        elif face_detector is not None:
            self._face_detector = face_detector
        else:
            self._face_detector = dlib.get_frontal_face_detector()
        self._face_detection_gap = 0xFFFFFFFF

    @property
    def face_detector(self):
        return self._face_detector

    @face_detector.setter
    def face_detector(self, detector):
        assert detector is not None
        self._face_detector = detector

    def reset(self, reset_face_detection_countdown=True):
        super(ZenoFaceTracker, self).reset(reset_face_detection_countdown)
        if reset_face_detection_countdown:
            self.reset_face_detection_countdown()

    def reset_face_detection_countdown(self):
        self._face_detection_gap = 0xFFFFFFFF

    def track(self, frame, target_face, use_bgr_colour_model=True):
        self.reset_face_detection_countdown()
        return super(ZenoFaceTracker, self).track(frame, target_face, use_bgr_colour_model=use_bgr_colour_model)

    def track(self, frame, redetect_face=False, use_bgr_colour_model=True):
        # About face detection
        if redetect_face:
            self.reset()
        if self._face_detection_gap < 0xFFFFFFFF:
            self._face_detection_gap += 1
        if not self.has_facial_landmarks:
            if self._face_detection_gap <= self.minimum_face_detection_gap:
                return self.current_state
            else:
                self._face_detection_gap = 0

        # Convert the frame to grayscale
        if frame.ndim == 3 and frame.shape[2] == 3:
            if use_bgr_colour_model:
                grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            grayscale_frame = frame

        # The actual processing
        if self.has_facial_landmarks:
            return super(ZenoFaceTracker, self).track(grayscale_frame, use_bgr_colour_model=use_bgr_colour_model)
        else:
            # Face detection
            frame_size = (grayscale_frame.shape[1], grayscale_frame.shape[0])
            face_detection_frame_size = (int(max(round(frame_size[0] * self.face_detection_scale), 1)),
                                         int(max(round(frame_size[1] * self.face_detection_scale), 1)))
            if face_detection_frame_size == frame_size:
                face_detection_frame = grayscale_frame
            else:
                face_detection_frame = cv2.resize(grayscale_frame, face_detection_frame_size)
            detected_faces = sorted([dlib.rectangle(int(round(face_box.left() / self.face_detection_scale)),
                                                    int(round(face_box.top() / self.face_detection_scale)),
                                                    int(round(face_box.right() / self.face_detection_scale)),
                                                    int(round(face_box.bottom() / self.face_detection_scale)))
                                     for face_box in self._face_detector(face_detection_frame)],
                                    key=dlib.rectangle.area, reverse=True)
            if (len(detected_faces) > 0 and detected_faces[0].width() >= self.minimum_face_size and
                    detected_faces[0].height() >= self.minimum_face_size):
                return super(ZenoFaceTracker, self).track(frame, (detected_faces[0].left(),
                                                                  detected_faces[0].top(),
                                                                  detected_faces[0].width(),
                                                                  detected_faces[0].height()),
                                                          use_bgr_colour_model=use_bgr_colour_model)
            else:
                return self.current_state
