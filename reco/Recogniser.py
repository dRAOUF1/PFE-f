import numpy as np
import cv2

import os
class Recogniser:

    """
    Class for face recognition using a pre-trained model.

    Args:
        modelPath (str): Path to the pre-trained model.
        disType (int, optional): Distance type for matching. 0 for cosine similarity, 1 for Norm-L2 distance. Defaults to 0.
    """

    def __init__(self, modelPath, disType=0):
        try:
            self._modelPath = modelPath
            self._model = cv2.FaceRecognizerSF.create(
                model=self._modelPath,
                config="",
                backend_id=0,
                target_id=0)

            self._disType = disType # 0: cosine similarity, 1: Norm-L2 distance
            assert self._disType in [0, 1], "0: Cosine similarity, 1: norm-L2 distance, others: invalid"

            self._threshold_cosine = float(os.getenv('RECOGNITION_THRESHOLD'))
            self._threshold_norml2 = 1.128
        except Exception as e:
            print(f"Error occurred during initialization: {str(e)}")

    def _preprocess(self, image, bbox):
        """
        Preprocesses the input image by aligning and cropping it based on the bounding box.

        Args:
            image (numpy.ndarray): Input image.
            bbox (tuple, optional): Bounding box coordinates. Defaults to None.

        Returns:
            numpy.ndarray: Preprocessed image.
        """
        try:
            if bbox is None:
                return image
            else:
                return self._model.alignCrop(image, bbox)
        except Exception as e:
            print(f"Error occurred during preprocessing: {str(e)}")
            return None

    def infer(self, image, bbox=None):
        """
        Performs inference on the input image.

        Args:
            image (numpy.ndarray): Input image.
            bbox (tuple, optional): Bounding box coordinates. Defaults to None.

        Returns:
            numpy.ndarray: Features extracted from the image.
        """
        try:
            # Preprocess
            inputBlob = self._preprocess(image, bbox)
            if inputBlob is None:
                return None
            # Forward
            features = self._model.feature(inputBlob)
            return features
        except Exception as e:
            print(f"Error occurred during inference: {str(e)}")
            return None

    def match(self, image1, face1, image2, face2):
        """
        Matches two faces based on their features extracted from the input images.

        Args:
            image1 (numpy.ndarray): First input image.
            face1 (tuple): Bounding box coordinates of the first face in image1.
            image2 (numpy.ndarray): Second input image.
            face2 (tuple): Bounding box coordinates of the second face in image2.

        Returns:
            float: Similarity score between the two faces. Returns 0 if the score is below the threshold.
        """
        try:
            feature1 = self.infer(image1, face1)
            feature2 = self.infer(image2, face2)

            if feature1 is None or feature2 is None:
                return 0

            if self._disType == 0: # COSINE
                cosine_score = self._model.match(feature1, feature2, self._disType)
                return cosine_score if cosine_score >= self._threshold_cosine else 0
            else: # NORM_L2
                norml2_distance = self._model.match(feature1, feature2, self._disType)
                return norml2_distance if norml2_distance <= self._threshold_norml2 else 0
        except Exception as e:
            print(f"Error occurred during matching: {str(e)}")
            return 0

    def match(self, feature1, feature2):
        """
        Matches two faces based on their features.

        Args:
            feature1 (numpy.ndarray): Features of the first face.
            feature2 (numpy.ndarray): Features of the second face.

        Returns:
            float: Similarity score between the two faces. Returns 0 if the score is below the threshold.
        """
        try:
            #verify type of feature1 and feature2
            assert type(feature1) == np.ndarray, "Invalid type for feature1"
            assert type(feature2) == np.ndarray, "Invalid type for feature2"
            
            if self._disType == 0: # COSINE
                cosine_score = self._model.match(feature1, feature2, self._disType)
                return cosine_score if cosine_score >= self._threshold_cosine else 0
            else: # NORM_L2
                norml2_distance = self._model.match(feature1, feature2, self._disType)
                return norml2_distance if norml2_distance <= self._threshold_norml2 else 0
        except Exception as e:
            print(f"Error occurred during matching: {str(e)}")
            return 0

