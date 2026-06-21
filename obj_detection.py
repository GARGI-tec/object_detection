import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
# Function to start the object detection process
def start_detection():
    root.destroy()  # Close the starting page window
    # Object detection code goes here...

# Create a Tkinter window for the starting page
root = tk.Tk()
root.title("Object Detection Application")

# Function to show an about dialog box
def show_about():
    messagebox.showinfo("About", "This application detects objects using a webcam.")
# Load background image
bg_image = Image.open("C:\\Users\\hp\\Downloads\\think_pic.jpeg")  # Provide the path to your background image

# Resize the image to fit the window size
bg_image = bg_image.resize((root.winfo_screenwidth(), root.winfo_screenheight()), Image.ANTIALIAS)

bg_photo = ImageTk.PhotoImage(bg_image)
bg_label = tk.Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Create a label for the company name
company_label = tk.Label(root, text="THINK RECYCLING", font=("Helvetica", 30, "bold"), fg="green")
company_label.pack(pady=20)

# Create a start button to initiate object detection
start_button = tk.Button(root, text="Start", command=start_detection, bg="Green", fg="white", font=("Helvetica", 20, "bold"))
start_button.pack(pady=10)

# Create a menu bar
menubar = tk.Menu(root)
helpmenu = tk.Menu(menubar, tearoff=0)
helpmenu.add_command(label="About", command=show_about)
menubar.add_cascade(label="Help", menu=helpmenu)
root.config(menu=menubar)

# Display the starting page window
root.mainloop()

thres = 0.5 # Threshold to detect object
nms_threshold = 0.2 #(0.1 to 1) 1 means no suppress , 0.1 means high suppress 
cap = cv2.VideoCapture(0) # Use 0 for built-in webcam, 1 for external webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH,800) #width 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,600) #height 
cap.set(cv2.CAP_PROP_BRIGHTNESS,100) #brightness 

classNames = []
with open('objects.txt','r') as f:
    classNames = f.read().splitlines()
print(classNames)

font = cv2.FONT_HERSHEY_PLAIN
#font = cv2.FONT_HERSHEY_COMPLEX
Colors = np.random.uniform(0, 255, size=(len(classNames), 3))

weightsPath = 'frozen_inference_graph.pb'
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
net = cv2.dnn_DetectionModel('frozen_inference_graph.pb','ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))

    indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
    print("Indices:", indices) # print to debug
    print("Type of indices:", type(indices)) # print to debug
    
    if len(classIds) != 0:
        for i in indices:
            box = bbox[i]
            color = Colors[classIds[i]-1]
            confidence = str(round(confs[i],2))
            x,y,w,h = box[0],box[1],box[2],box[3]
            cv2.rectangle(img, (x,y), (x+w,y+h), color, thickness=2)
            cv2.putText(img, classNames[classIds[i]-1]+" "+confidence,(x+10,y+20),
                        font,1,color,2)


    cv2.imshow("ObjectDetection",img)
    if cv2.waitKey(1) & 0xFF == ord('q'): # Add a condition to quit the loop
        break
cap.release()
cv2.destroyAllWindows()


def main():
    # Open the camera
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('Camera', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera
    cap.release()
    cv2.destroyAllWindows()

if _name_ == "main":
    main()
