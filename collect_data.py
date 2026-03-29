import cv2
import os

label = input("Enter finger number (0-5): ")

path = f"finger_dataset/{label}"
os.makedirs(path, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0

while True:
    ret, frame = cap.read()

    roi = frame[100:400, 100:400]
    cv2.rectangle(frame,(100,100),(400,400),(0,255,0),2)

    cv2.imshow("ROI", roi)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        file = f"{path}/{count}.jpg"
        cv2.imwrite(file, roi)
        count += 1
        print("saved",count)

    if key == 27:
        break
    
    
cap.release()
cv2.destroyAllWindows()