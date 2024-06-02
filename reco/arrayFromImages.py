import os
import cv2
import pymongo
import sys
import getopt


if __name__ == "__main__":
    """
    This script reads images from a specified directory, detects faces in the images,
    extracts facial embeddings, and stores them in a MongoDB database.
    """

    # Get the path to the images directory and the MongoDB URL from command-line arguments
    try:
        IMAGES_PATH = sys.argv[1]
        MONGO_URL = sys.argv[2]
        FACE_DETECTOR = sys.argv[3]
        FACE_RECOGNIZER = sys.argv[4]
    except IndexError:
        print("Error: Please provide the path to the images directory and the MongoDB URL as command-line arguments.")
        sys.exit(1)

    try:
        # Connect to the MongoDB database
        client = pymongo.MongoClient(MONGO_URL)
        db = client["mydb"]
    except (pymongo.errors.ConnectionError,pymongo.errors.ServerSelectionTimeoutError):
        print("Error: Failed to connect to the MongoDB database.")
        sys.exit(1)

    # Create a face detector and a face recognizer
    try:
        detector = cv2.FaceDetectorYN.create(
            model=FACE_DETECTOR,
            config="",
            input_size=[320, 320],
            score_threshold=0.65,
            nms_threshold=0.3,
            top_k=5000,
            backend_id=0,
            target_id=0)
        recognizer = cv2.FaceRecognizerSF.create(
            model=FACE_RECOGNIZER,
            config="",
            backend_id=0,
            target_id=0)
    except cv2.error as e:
        print(f"Error: Failed to create face detector or recognizer. {e}")
        sys.exit(1)

    # Get the paths of all image files in the specified directory
    images = [os.path.join(dossier, fichier) for dossier, sous_dossiers, fichiers in os.walk(IMAGES_PATH) for fichier in fichiers if fichier.endswith('.jpg') or fichier.endswith('.png') or fichier.endswith('.jpeg')]
    etudiants_collection = db['etudiants']
    matricules_etudiants = etudiants_collection.distinct('MatriculeEtd')
    # Process each image
    for image in images:
        # try:
        img = cv2.imread(image)
        detector.setInputSize([img.shape[1], img.shape[0]])
        faces = detector.detect(img)
        if len(faces) > 0:
            inputBlob = recognizer.alignCrop(img, faces[1][:-1])
            embedding = recognizer.feature(inputBlob)
            matricule = image.split('/')[-1].split('\\')[-1].split('.')[0]
            etudiant = {"MatriculeEtd": matricule, "embedding": embedding.tolist()}
            try:
                if (matricule not in matricules_etudiants):
                    print(f"Error: Matricule {matricule} not found in the etudiants collection.")
                    continue
                # Insert the student's embedding into the MongoDB database
                result = db.embeddings.insert_one(etudiant)
            except pymongo.errors.DuplicateKeyError:
                print("Error: Duplicate key.")
                continue
        # except cv2.error as e:
            # print(f"Error: Failed to process image {image}. {e}")
