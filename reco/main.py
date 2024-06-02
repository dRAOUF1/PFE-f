from App import App
import cv2 ,time, os
import requests
import sys,getopt
from centroidtracker2 import CentroidTracker
from VideoStream import VideoStream
from dotenv import load_dotenv

load_dotenv()



rotations = {0:(None,(640,360 )),
             90:(cv2.ROTATE_90_CLOCKWISE,(360,640 )),
             180:(cv2.ROTATE_180,(640,360 )),
             270:(cv2.ROTATE_90_COUNTERCLOCKWISE,(360,640 ))
    }

if __name__ == '__main__':
    adresse_ip = os.getenv('URL_BACKEND')
    #port = 3001
    degre_rotation = int(os.getenv('ROTATION'))
    save_faces = False

    # Analyser les arguments de la ligne de commande
   

    # # VÃ©rifier le degrÃ© de rotation
    # if degre_rotation not in (0, 90, 180, 270):
    #     print("Degré de rotation invalide : {}".format(degre_rotation))
    #     sys.exit(1)
    # else:
    #     degre_rotation = rotations[degre_rotation]
    # if adresse_ip == None:
    #     print("Adresse ip invalide")
    #     sys.exit(1)
    # if port == None:
    #     print("port invalide")
    #     sys.exit(1)
    degre_rotation = rotations[180]
    ct = CentroidTracker()
    app = App(f"http://{adresse_ip}/getEmbeddings/all/all")
    print('lol')
    print("App embedings",len(app.embeddings))
    print("fin app build")
    videostream = VideoStream(record=True)
    videostream.start()
    #cap = cv2.VideoCapture("C:/Users/yas/Desktop/test.avi")

    prev_frame_time = 0
    fp = []
    c = 0
    print("debut boucle")
    while True:
        rects = []
        frame=videostream.read()
#         if not ret:
#             break
        
#         if not degre_rotation[0] == None:
#             frame = cv2.rotate(frame, degre_rotation[0])
        
        c+=1
        if frame is None:
            continue
        # frame = cv2.resize(frame,(128,96))
        if c%2==0:
            faces = app.find_match(frame)

            for face in faces:
                if save_faces:
                    cropped = frame[int(face.face.y1):int(face.face.y2), int(face.face.x1):int(face.face.x2)]
                    cropped = cv2.resize(cropped,(128,128))
                    cv2.imwrite(f"faces/{face.name}-{time.time()}.jpg",cropped)

                rects.append((int(face.face.x1), int(face.face.y1), int(face.face.x2), int(face.face.y2), face.name))
                frame = app.Draw(frame,face)
                #print("\n",face.name,matricule[face.name] )
                print(face.name)    
            
                r = requests.post(f"http://{adresse_ip}/postEtdsPresent",json={"matricule":face.name})
                print(r)
            objects = ct.update(rects)
                    #exit(0)

        
        new_frame_time = time.time()
        try: 
            fps = 1/(new_frame_time-prev_frame_time) 
        except (ZeroDivisionError):
            pass
        prev_frame_time = new_frame_time 
        fp.append(fps)
        fps = str(int(fps))
        #cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
        try:
            cv2.imshow("frame",frame)
            k = cv2.waitKey(10)
            if k == 27:         # wait for ESC key to exit
                break
        except:
            print('erreur')
            

    #average of fp
    c = 0
    for f in fp:
        c+=f
    try:
        print("average",c/len(fp))
    except:
        pass
    cv2.destroyAllWindows()
    

