import cv2
from ultralytics import YOLO
from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename

main = tkinter.Tk()
main.title("Weapon Detection")
main.geometry("1300x900")

image_path = None
video_path = None

def detect_objects_in_photo(image_path):
    image_orig = cv2.imread(image_path)
    
    yolo_model = YOLO('./runs/detect/Normal_Compressed/weights/best.pt')
    
    results = yolo_model(image_orig)

    for result in results:
        classes = result.names
        cls = result.boxes.cls
        conf = result.boxes.conf
        detections = result.boxes.xyxy

        for pos, detection in enumerate(detections):
            if conf[pos] >= 0.5:
                xmin, ymin, xmax, ymax = detection
                label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}" 
                color = (0, int(cls[pos]), 255)
                cv2.rectangle(image_orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                cv2.putText(image_orig, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 1, cv2.LINE_AA)
    
    #result_path = "./imgs/Test/teste.jpg"
    #cv2.imwrite(result_path, image_orig)

    cv2.imshow("Detected Objects", image_orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #return result_path

def detect_objects_in_video():
    yolo_model = YOLO('./runs/detect/Normal_Compressed/weights/best.pt')
    video_capture = cv2.VideoCapture(video_path)
    #width = int(video_capture.get(3))
    #height = int(video_capture.get(4))
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #result_video_path = "detected_objects_video2.avi"
    #out = cv2.VideoWriter(result_video_path, fourcc, 20.0, (width, height))

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        results = yolo_model(frame)

        for result in results:
            classes = result.names
            cls = result.boxes.cls
            conf = result.boxes.conf
            detections = result.boxes.xyxy

            for pos, detection in enumerate(detections):
                if conf[pos] >= 0.5:
                    xmin, ymin, xmax, ymax = detection
                    label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}" 
                    color = (0, int(cls[pos]), 255)
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                    cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        cv2.imshow("Image", frame)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break 
        #out.write(frame)
    video_capture.release()
    #out.release()

    #return result_video_path

def uploadImage():
    global image_path
    image_path = askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if image_path:
        detect_objects_in_photo(image_path)

def uploadVideo():
    global video_path
    video_path = askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
    if video_path:
        detect_objects_in_video()

font = ('times', 24, 'bold')
title = Label(main, text='Weapon Detection',anchor=CENTER, justify=CENTER)
title.config(bg='black', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 40, 'bold')

uploadButton = Button(main, text="🏞️ Detect Weapon from Image", command=uploadImage)
uploadButton.place(x=400,y=200)
uploadButton.config(font=font1)

videoButton = Button(main, text="🎥 Detect Weapon from Video", command=uploadVideo)
videoButton.place(x=400,y=400)
videoButton.config(font=font1)

main.config(bg='#ffffff')
main.mainloop()