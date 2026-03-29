import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

model = load_model("finger_model.h5")

cap = cv2.VideoCapture(0)

prev_time = 0

while True:

    ret, frame = cap.read()

    roi = frame[100:400,100:400]

    img = cv2.resize(roi,(64,64))
    img = img/255.0
    img = np.reshape(img,(1,64,64,3))

    pred = model.predict(img)
    finger = np.argmax(pred)

    # FPS calculation
    curr_time = time.time()
    fps = 1/(curr_time-prev_time)
    prev_time = curr_time

    fps = int(fps)

    # display results
    cv2.putText(frame,f"Fingers: {finger}",(50,50),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.putText(frame,f"FPS: {fps}",(50,100),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

    cv2.rectangle(frame,(100,100),(400,400),(0,255,0),2)

    cv2.imshow("Finger Detection",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()