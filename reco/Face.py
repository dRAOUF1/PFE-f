from typing import List, Tuple, Optional
import numpy as np
class Face:
    def __init__(
        self,
        x,
        y,
        x2,
        y2,
        confidence,
        left_eye,
        right_eye,
        nose,
        left_mouth,
        right_mouth
    ):
        """
        Initializes a Face object with the given parameters.

        Args:
            x (int): The x-coordinate of the top-left corner of the face bounding box.
            y (int): The y-coordinate of the top-left corner of the face bounding box.
            x2 (int): The x-coordinate of the bottom-right corner of the face bounding box.
            y2 (int): The y-coordinate of the bottom-right corner of the face bounding box.
            confidence (float): The confidence score of the face detection.
            left_eye (tuple, optional): The coordinates of the left eye. Defaults to None.
            right_eye (tuple, optional): The coordinates of the right eye. Defaults to None.
            nose (tuple, optional): The coordinates of the nose. Defaults to None.
            left_mouth (tuple, optional): The coordinates of the left mouth corner. Defaults to None.
            right_mouth (tuple, optional): The coordinates of the right mouth corner. Defaults to None.
        """
        self.x1 = x
        self.y1 = y
        self.x2 = x2
        self.y2 = y2
        self.left_eye = left_eye
        self.right_eye = right_eye
        self.nose = nose
        self.left_mouth = left_mouth
        self.right_mouth = right_mouth
        self.confidence = confidence


    def toArray(self) -> np.ndarray:
        """
        Converts the Face object to a numpy array.

        Returns:
            numpy.ndarray: The array representation of the Face object.
        """
        try:
            return np.array([self.x1, self.y1, self.x2, self.y2, self.right_eye[0], self.right_eye[1], self.left_eye[0], self.left_eye[1], self.nose[0], self.nose[1], self.right_mouth[0], self.right_mouth[1], self.left_mouth[0], self.left_mouth[1], self.confidence])
        except (TypeError, IndexError):
            raise ValueError("Invalid face object. Some attributes are missing or have incorrect format.")

    def setId(self, id: int):
        """
        Sets the ID of the face.

        Args:
            id (int): The ID to set.
        """
        self.id = id