import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# -- firebase settings --

cred = credentials.Certificate('Firebase_JSON/firebase_key.json')
firebase_admin.initialize_app(cred,{
    'databaseURL':'https://sign-detector-0101-default-rtdb.firebaseio.com/'
    })

# -- loading yolo model --

net = cv2.dnn.readNetFromDarknet('yolov4-tiny_custom_ptult.cfg', 'yolov4-tiny_custom_final_ultimu.weights')
cap = cv2.VideoCapture(0)

classes = ['drum_prioritate', 'stop','trecere_pietoni','atentie_koala']
    
prev_frame_time = 0
  
new_frame_time = 0    

while True:
    _, frame = cap.read()
    ht, wt, _ = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), (0,0,0), swapRB = True, crop = False)

    net.setInput(blob)

    last_layer = net.getUnconnectedOutLayersNames()
    layer_out = net.forward(last_layer)

    boxes = []
    confidences = []
    class_ids = []
    
    

    for output in layer_out:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > .6:
                center_x = int(detection[0] * wt )
                center_y = int(detection[1] * ht )
                w = int(detection[2] * wt )
                h = int(detection[3] * ht )
                
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                
                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
                
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, .5, .4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size = (len(boxes),3))
    if  len(indexes)>0:
        for i in indexes.flatten():
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            timestamp = int(time.time())
            cv2.putText(frame, label + " "+ confidence, (x,y+20), font, 2, (255, 255, 255), 2)
            if(label and not label.isspace()):
                ref=db.reference('Detections')
                push_ref = ref.push({
                    'detection':label,
                    'timestamp': timestamp
                    })
                print("sent to firebase - "+label + " "+ str(timestamp))
    new_frame_time = time.time()

    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
  
    fps = float(fps)
  
    fps = str(round(fps,2))
  
    cv2.putText(frame, "FPS: "+fps, (7, 70), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Sign Detector' , frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()