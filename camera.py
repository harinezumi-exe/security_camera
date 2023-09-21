import cv2
import time
import datetime


address = input("Enter FULL camera address: ")

capture = cv2.VideoCapture(address)

# laptop camera
# capture = cv2.VideoCapture(0)

# phone camera example ip
# capture = cv2.VideoCapture("https://172.16.1.66:8080/video")

# classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

detection = False
detection_stopped_time = None
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 5

frame_size = (int(capture.get(3)), int(capture.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

while True:
    _, frame = capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    '''
    detectMultiScale(image, 
                    scaleFactor: [1.1-1.5] higher = faster lower = more accurate),
                    minNumOfNeighbours aka how many overlapping frames to call it
                        a face, higher = slower, less faces; lower = faster,
                        possible it will detect things that aren't faces
    '''
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bodies = body_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) + len(bodies) >0:
        if detection:
            timer_started = False
        else:
            detection = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 20, frame_size)
            print("Started recording!")
    elif detection:
        if timer_started:
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection = False
                timer_started = False
                out.release()
                print("Stopped Recording!")
        else:
            timer_started = True
            detection_stopped_time = time.time()
    
    if detection:
        out.write(frame)

    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 3) # BGR not RGB
    
    cv2.imshow("Camera", frame)

    # this one is not necessary, program works faster if it doesn't have to display the video
    if cv2.waitKey(1) == ord('q'):
        break

out.release()
capture.release()
cv2.destroyAllWindows()