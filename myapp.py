import tkinter as tk
from tkinter.ttk import *
import cv2
from PIL import Image, ImageTk
from tkinter import filedialog
from ultralytics import YOLO
import pandas as pd
import cvzone

# Global variables for OpenCV-related objects and flags
cap = None
is_camera_on = False
frame_count = 0
frame_skip_threshold = 3
video_paused = False
path1 = "./models/bestfire.pt"
model = YOLO(path1)


# Function to read classes from a file
def read_classes_from_file(file_path):
    with open(file_path, 'r') as file:
        classes = [line.strip() for line in file]
    return classes


# Function to start the webcam feed
def start_webcam():
    global cap, is_camera_on, video_paused
    if not is_camera_on:
        cap = cv2.VideoCapture(0)  # Use the default webcam (you can change the index if needed)
        is_camera_on = True
        video_paused = False
        update_canvas()  # Start updating the canvas


# Function to stop the webcam feed
def stop_webcam():
    global cap, is_camera_on, video_paused
    if cap is not None:
        cap.release()
        is_camera_on = False
        video_paused = False


# Function to pause or resume the video
def pause_resume_video():
    global video_paused
    video_paused = not video_paused


# Function to start video playback from a file
def select_file():
    global cap, is_camera_on, video_paused
    if is_camera_on:
        stop_webcam()  # Stop the webcam feed if running
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if file_path:
        cap = cv2.VideoCapture(file_path)
        is_camera_on = True
        video_paused = False
        update_canvas()  # Start updating the canvas with the video


# Select the model to work with
def selectmodel(*args):
    global path1, model, class_selection, class_list
    name = model_selection.get()
    path1 = "./models/" + name + ".pt"
    model = YOLO(path1)
    if name == "bestfire":
        class_list = read_classes_from_file('fireSmoke.txt')
    else:
        class_list = read_classes_from_file('coco.txt')
    class_selection.set("All")
    class_selection_menu = class_selection_entry["menu"]
    class_selection_menu.delete(0, "end")
    for item in class_list:
        class_selection_menu.add_command(label=item, command=tk._setit(class_selection, item))


# Function to update the Canvas with the webcam frame or video frame
def update_canvas():
    global is_camera_on, frame_count, video_paused, model
    if is_camera_on:
        if not video_paused:
            ret, frame = cap.read()
            if ret:
                frame_count += 1
                if frame_count % frame_skip_threshold != 0:
                    canvas.after(10, update_canvas)
                    return

                frame = cv2.resize(frame, (1020, 500))
                selected_class = class_selection.get()

                results = model.track(source=frame, conf=0.4, persist=True, tracker="bytetrack.yaml")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                a = results[0].boxes.data
                px = pd.DataFrame(a).astype("float")
                print(px.head())
                for index, row in px.iterrows():
                    x1 = int(row[0])
                    y1 = int(row[1])
                    x2 = int(row[2])
                    y2 = int(row[3])
                    t = int(row[4])
                    if len(row) == 7:
                        p = row[5]
                        d = int(row[6])
                        c = class_list[d]
                        p = "{:.2f}".format(p)
                        if selected_class == "All" or c == selected_class:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                            cvzone.putTextRect(frame, f'id:{t} {c} {p}', (x1, y1+20), 1, 1, (255, 255, 255), (255, 0, 0))
                    else:
                        p = row[4]
                        d = int(row[5])
                        c = class_list[d]
                        p = "{:.2f}".format(p)
                        if selected_class == "All" or c == selected_class:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                            cvzone.putTextRect(frame, f'{c} {p}', (x1, y1+20), 1, 1, (255, 255, 255), (255, 0, 0))

                photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                canvas.img = photo
                canvas.create_image(0, 0, anchor=tk.NW, image=photo)

        canvas.after(10, update_canvas)


# Function to quit the application
def quit_app():
    stop_webcam()
    root.quit()
    root.destroy()


# Create the main Tkinter window
root = tk.Tk()
root.title("MIAAD Detection Tracking YOLOv8")
root.iconbitmap('icon.ico')
root.wm_iconbitmap('icon.ico')

# Create a Canvas widget to display the webcam feed or video
canvas = tk.Canvas(root, width=1020, height=500)
canvas.pack(fill='both', expand=True)

# Define your class list and model list using the appropriate files

model_list = read_classes_from_file('modelslist.txt')

model_selection = tk.StringVar()
model_selection.set("bestfire")  # Default selection is "bestfire"
model_selection_label = tk.Label(root, text="Select Model:")
model_selection_label.pack(side='left')
model_selection.trace('w', selectmodel)  # Monitor changes to the model selection
model_selection_entry = tk.OptionMenu(root, model_selection, "bestfire", *model_list)
model_selection_entry.pack(side='left')

class_list = read_classes_from_file('fireSmoke.txt')  # Default class list

# Create dropdown for selecting classes
class_selection = tk.StringVar()
class_selection.set("All")  # Default selection is "All"
class_selection_label = tk.Label(root, text="Select Class:")
class_selection_label.pack(side='left')
class_selection_entry = tk.OptionMenu(root, class_selection, "All", *class_list)
class_selection_entry.pack(side='left')

# Create a frame to hold the buttons
button_frame = tk.Frame(root)
button_frame.pack(fill='x')

# Create a "Select File" button to choose a video file
file_button = tk.Button(button_frame, text="Select File", command=select_file)
file_button.pack(side='left')

# Create a "Real-time" button to start the webcam feed
play_button = tk.Button(button_frame, text="Real-time", command=start_webcam)
play_button.pack(side='left')

# Create a "Pause/Resume" button to pause or resume video
pause_button = tk.Button(button_frame, text="Pause/Resume", command=pause_resume_video, background="orange")
pause_button.pack(side='left')

# Create a "Stop" button to stop the webcam feed
stop_button = tk.Button(button_frame, text="Stop", command=stop_webcam, background="red")
stop_button.pack(side='left')

# Display an initial image on the canvas (replace 'background.jpg' with your image path)
initial_image = Image.open('background.jpg')  # Replace 'background.jpg' with your image path
initial_photo = ImageTk.PhotoImage(image=initial_image)
canvas.img = initial_photo
canvas.create_image(0, 0, anchor=tk.NW, image=initial_photo)

# Start the Tkinter main loop
root.mainloop()
