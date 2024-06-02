import numpy as np
import cv2
from Face import Face

class Detector:
    """
    A class that represents a face detector.

    Attributes:
        _modelPath (str): The path to the face detection model.
        _inputSize (tuple): The input size of the model in the format (width, height).
        _confThreshold (float): The confidence threshold for face detection.
        _nmsThreshold (float): The non-maximum suppression threshold for face detection.
        _model (cv2.FaceDetectorYN): The face detection model.


    Methods:
        __init__(self, modelPath, inputSize=[320, 320], confThreshold=0.6, nmsThreshold=0.3):
            Initializes the Detector object with the specified parameters.
        setInputSize(self, input_size):
            Sets the input size of the model.
        infer(self, image):
            Performs face detection on the given image and returns a list of detected faces.
    """

    def __init__(self, modelPath, inputSize=[320, 320], confThreshold=0.6, nmsThreshold=0.3):
        """
        Initializes the Detector object with the specified parameters.

        Args:
            modelPath (str): The path to the face detection model.
            inputSize (list, optional): The input size of the model in the format [width, height]. Defaults to [320, 320].
            confThreshold (float, optional): The confidence threshold for face detection. Defaults to 0.6.
            nmsThreshold (float, optional): The non-maximum suppression threshold for face detection. Defaults to 0.3.
        """
        self._modelPath = modelPath
        self._inputSize = tuple(inputSize) # [w, h]
        self._confThreshold = confThreshold
        self._nmsThreshold = nmsThreshold

        self._model = cv2.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=5000,
            backend_id=0,
            target_id=0)


    def setInputSize(self, input_size):
        """
        Sets the input size of the model.

        Args:
            input_size (list): The input size of the model in the format [width, height].
        """
        self._model.setInputSize(tuple(input_size))

    def infer(self, image):
        """
        Performs face detection on the given image and returns a list of detected faces.

        Args:
            image: The image on which to perform face detection.

        Returns:
            list: A list of detected faces, where each face is represented as a Face object.
        """
        # Forward
        results = self._model.detect(image)
        if results[1] is not None:
            faces = [ Face(
                x=face[0],
                y=face[1],
                x2=face[0] + face[2],
                y2=face[1] + face[3],
                confidence=face[14],
                right_eye=(face[4], face[5]),
                left_eye=(face[6], face[7]),
                nose=(face[8], face[9]),
                right_mouth=(face[10], face[11]), # (x, y)
                left_mouth=(face[12], face[13]) # (x, y
            ) for face in results[1] if face[2]>27 and face[3]>27]
            return faces
        
        return []

    