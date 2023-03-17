#PARAMETERS
MODEL_PATH = r"C:\\Users\\Kayra\\Desktop\\yolo8best.pt" #Path of the model
CENTER_AREA_OFFSET = 50 #Size of the center area


import cv2
import ultralytics
from ultralytics import YOLO

model = YOLO(MODEL_PATH)


vid = cv2.VideoCapture(0)
#If resolutiom change is needed
#vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ret, frame = vid.read()
(h, w) = frame.shape[:2]
centerY = int(h//2)
centerX = int(w//2)
p1Area = (centerX-CENTER_AREA_OFFSET, centerY-CENTER_AREA_OFFSET)
p2Area = (centerX+CENTER_AREA_OFFSET,centerY+CENTER_AREA_OFFSET)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
  
    #Model Detections
    result = model.predict(frame)
    detection = result[0].boxes.xyxy.cpu().numpy()
    if len(detection) == 1: #If there is a detection, get coordinates 
        x1 = int(detection[0][0])
        y1 = int(detection[0][1])
        x2 = int(detection[0][2])
        y2 = int(detection[0][3])
        detectionCenter = ((x1+x2)//2, (y1+y2)//2) #THIS IS THE CENTER POINT OF DETECTION RECTANGLE
        detectionCenter1 = ((x1+x2)//2-5, (y1+y2)//2-5)
        detectionCenter2 = ((x1+x2)//2+5, (y1+y2)//2+5)

        
        

        #Check if detection center is inside determined center area
        if p1Area[0]<detectionCenter[0] and p1Area[1]<detectionCenter[1] and p2Area[0]>detectionCenter[0] and p2Area[1]>detectionCenter[1]:
            frame = cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 0), 2)#Draw detection box
            frame = cv2.rectangle(frame, detectionCenter1, detectionCenter2, (0, 255, 0), -1)#Draw detection center box
        else:
            frame = cv2.rectangle(frame, (x1,y1), (x2,y2), (255, 0, 0), 2)#Draw detection box
            frame = cv2.rectangle(frame, detectionCenter1, detectionCenter2, (255, 0, 0), -1)#Draw detection center box



    # Display the resulting frame


    frame = cv2.rectangle(frame, p1Area, p2Area, (0, 0, 255), 1)#Draw center box
    cv2.imshow('frame', frame)
    
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()