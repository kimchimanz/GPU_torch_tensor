import cv2
import Filter as ft
from AIDetector_pytorch import Detector

webcam = cv2.VideoCapture(0)
det = Detector()
func_status = {}

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

while webcam.isOpened():
    status, frame = webcam.read()

    frame = ft.yolo(frame)
    det.feedCap(frame, func_status)
    if status:
        cv2.imshow("test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()