from Detector import Detector
from Recogniser import Recogniser
from RecognizedFace import RecognizedFace
from Face import Face
import random
import os
import cv2
import requests
import numpy as np
from dotenv import load_dotenv
load_dotenv()

class App:
    """
    Class representing an application for face recognition and detection.

    Args:
        db_path (str): The path to the directory containing the face images for building the embeddings.
        recognition_model_path (str): The path to the face recognition model file.
        detection_model_path (str): The path to the face detection model file.
    """
    def __init__(self, db_path, recognition_model_path="/home/pi/Desktop/reco/models/face_recognition_sface_2021dec.onnx", detection_model_path='/home/pi/Desktop/reco/models/face_detection_yunet_2023mar.onnx'):
        """
        Initializes an instance of the App class.

        Args:
            db_path (str): The path to the directory containing the face images for building the embeddings.
            recognition_model_path (str): The path to the face recognition model file.
            detection_model_path (str): The path to the face detection model file.
        """
        try:
            self.detector = Detector(modelPath=detection_model_path,
                                     inputSize=[320, 320],
                                     confThreshold=float(os.getenv('DETECTION_THRESHOLD')),
                                     nmsThreshold=0.3,
                                    )
            self.recognizer = Recogniser(modelPath=recognition_model_path, disType=0)
            self.embeddings = self._getEmbeddingsFromBackend(db_path)
        except Exception as e:
            print(f"Error occurred during initialization: {str(e)}")

    
    def _getEmbeddingsFromBackend(self, db_Url):
        try:
            r = requests.get(db_Url)
        except Exception as e:
            print(f"Error occurred during getting embeddings from backend: {str(e)}")
            return {}
        # cast the embeddings to an array
        res = r.json()
        for key in res.keys():
            res[key] = np.array(res[key]).astype(np.float32)
        return res


    def localDbToEmbeddings(self, db):
        """
        Builds the embeddings dictionary from the face images in the specified directory.

        Args:
            db (str): The path to the directory containing the face images.

        Returns:
            dict: A dictionary mapping image names to their corresponding embeddings.
        """
        try:
            images = [os.path.join(dossier, fichier) for dossier, sous_dossiers, fichiers in os.walk(db) for fichier in fichiers if fichier.endswith('.jpg') or fichier.endswith('.png') or fichier.endswith('.jpeg')]
            embeddings = {im.split('/')[-1].split('\\')[-1].split('.')[0]: None for im in images}
            for image in images:
                img = cv2.imread(image)
                self.detector.setInputSize([img.shape[1], img.shape[0]])
                faces = self.detector.infer(img)
                if len(faces) > 0:
                    for face in faces:
                        embedding = self.recognizer.infer(img, face.toArray()[:-1])
                        embeddings[image.split('/')[-1].split('\\')[-1].split('.')[0]] = embedding
            return embeddings
        except Exception as e:
            print(f"Error occurred during building embeddings: {str(e)}")
            return {}

    def extractFaces(self, frame):
        """
        Extracts faces from the given frame using the face detection model.

        Args:
            frame (numpy.ndarray): The input frame.

        Returns:
            list: A list of detected faces.
        """
        try:
            self.detector.setInputSize([frame.shape[1], frame.shape[0]])
            faces = self.detector.infer(frame)
            return faces
        except Exception as e:
            print(f"Error occurred during face extraction: {str(e)}")
            return []

    def find_match(self, image):
        """
        Finds matches for the faces in the given image using the face recognition model.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            list: A list of recognized faces.
        """
        try:
            recognizedFaces = []
            self.detector.setInputSize([image.shape[1], image.shape[0]])
            faces = self.detector.infer(image)
            if len(faces) > 0:
                for face in faces:
                    minDist = 0
                    minKey = None
                    embedding = self.recognizer.infer(image, face.toArray()[:-1])
                    for key, value in self.embeddings.items():
                        if value is not None:
                            result = self.recognizer.match(embedding, value)
                            if result != 0:
                                if minDist == 0:
                                    minDist = result
                                    minKey = key
                                elif result > minDist:
                                    minDist = result
                                    minKey = key
                    if minKey is not None:
                        recognizedFaces.append(RecognizedFace(minKey, minDist, face))
            return recognizedFaces
        except Exception as e:
            print(f"Error occurred during face recognition: {str(e)}")
            return []

    def facesFromVideo(self, video_path, output_path):
        """
        Extracts faces from the specified video and saves them to the output directory.

        Args:
            video_path (str): The path to the input video file.
            output_path (str): The path to the output directory.
        """
        try:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            cap = cv2.VideoCapture(video_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                faces = self.find_match(frame)
                for face in faces:
                    cropped = frame[int(face.face.y1):int(face.face.y2), int(face.face.x1):int(face.face.x2)]
                    cropped = cv2.resize(cropped, (128, 128))
                    cv2.imwrite(f"{output_path}/{face.name}-{random.randint(0, 10000)}.jpg", cropped)
            cap.release()
        except Exception as e:
            print(f"Error occurred during face extraction from video: {str(e)}")

    def Draw(self, frame, obj, keypoints=False):
        """
        Draws the specified object on the given frame.

        Args:
            frame (numpy.ndarray): The input frame.
            obj (object): The object to be drawn.
            keypoints (bool, optional): Whether to draw keypoints. Defaults to False.

        Returns:
            numpy.ndarray: The frame with the object drawn on it.
        """
        try:
            if isinstance(obj, RecognizedFace):
                return self._draw_recognized_face(frame, obj, keypoints)
            elif isinstance(obj, Face):
                return self._draw_face(frame, obj, keypoints)
            else:
                raise ValueError("Object not supported.")
        except Exception as e:
            print(f"Error occurred during drawing: {str(e)}")
            return frame

    def _draw_recognized_face(self, frame, recoFace: RecognizedFace, keypoints=False):
        """
        Draws a recognized face on the given frame.

        Args:
            frame (numpy.ndarray): The input frame.
            recoFace (RecognizedFace): The recognized face object.
            keypoints (bool, optional): Whether to draw keypoints. Defaults to False.

        Returns:
            numpy.ndarray: The frame with the recognized face drawn on it.
        """
        try:
            frame = self._draw_face(frame, recoFace.face, keypoints)
            x1, y1, x2, y2 = int(recoFace.face.x1), int(recoFace.face.y1), int(recoFace.face.x2), int(recoFace.face.y2)
            frame = cv2.rectangle(frame, (x1, y2), (x2, y2 + 15), (255, 0, 0), -1)
            cv2.putText(frame, recoFace.name + f"///{recoFace.distance:.2f}", (x1-15, y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
            return frame
        except Exception as e:
            print(f"Error occurred during drawing recognized face: {str(e)}")
            return frame

    def _draw_face(self, frame, face: Face, keypoints=False):
        """
        Draws a face on the given frame.

        Args:
            frame (numpy.ndarray): The input frame.
            face (Face): The face object.
            keypoints (bool, optional): Whether to draw keypoints. Defaults to False.

        Returns:
            numpy.ndarray: The frame with the face drawn on it.
        """
        try:
            x1, y1, x2, y2 = int(face.x1), int(face.y1), int(face.x2), int(face.y2)
            confidence = face.confidence
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            frame = cv2.putText(frame, f"{confidence:.3f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if keypoints:
                left_eye = (int(face.left_eye[0]), int(face.left_eye[1]))
                right_eye = (int(face.right_eye[0]), int(face.right_eye[1]))
                nose = (int(face.nose[0]), int(face.nose[1]))
                left_mouth = (int(face.left_mouth[0]), int(face.left_mouth[1]))
                right_mouth = (int(face.right_mouth[0]), int(face.right_mouth[1]))
                frame = cv2.circle(frame, left_eye, 3, (0, 0, 255), 2)
                frame = cv2.circle(frame, right_eye, 3, (0, 0, 255), 2)
                frame = cv2.circle(frame, nose, 3, (255, 0, 0), 2)
                frame = cv2.circle(frame, left_mouth, 3, (0, 255, 255), 2)
                frame = cv2.circle(frame, right_mouth, 3, (0, 255, 255), 2)
            return frame
        except Exception as e:
            print(f"Error occurred during drawing face: {str(e)}")
            return frame

    def get_embedding(self,frame):
        """
        Extracts the embeddings from the given frame.

        Args:
            frame (numpy.ndarray): The input frame.

        Returns:
            numpy.ndarray: The embeddings extracted from the frame.
        """
        try:
            faces = self.extractFaces(frame)
            if len(faces) > 0:
                for face in faces:
                    embedding = self.recognizer.infer(frame, face.toArray()[:-1])
                    return embedding
            return None
        except Exception as e:
            print(f"Error occurred during getting embeddings: {str(e)}")
            return None