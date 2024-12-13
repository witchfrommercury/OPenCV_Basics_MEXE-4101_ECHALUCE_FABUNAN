# Echaluce_Fabunan_DrawingRectanglesOnCarsOnTheHighway

### Drawing Rectangles On Cars On The Highway

### 🚗 Introduction

- In conclusion, geometric shapes are very important for computer vision tasks that have to do with cars; they are the critical tool for researchers and practitioners. They give a visual and mathematical representation of vehicles in a way that closes the gap between raw visual data and actionable insights. It is either in bounding boxes, polygons, or keypoints, which help algorithms to understand, process, and analyze complex scenes involving cars.

- Their applications range from far and wide, driving innovations in high-tech fields such as autonomous driving, where accurate car detection and tracking are essential for safety and efficiency. In traffic management, these shapes help monitor vehicle movements, analyze congestion patterns, and enforce regulations. Similarly, in smart surveillance systems, they enhance the ability to identify vehicles, recognize license plates, and even detect suspicious activities.

- Geometric shapes also find an important place in the improvement of model interpretability, validation of predictions, and rigorous testing. These shapes give a tangible means of visualizing model outputs, thus pointing out errors or areas for improvement. By using them in data augmentation and edge case handling, these shapes contribute to making robust and versatile models capable of performing reliably in various real-world scenarios.Therefore, the use of geometric shapes in computer vision not only enhances understanding of visual data but also propels innovation across a multitude of sectors, thus making them essential in the pursuit of smarter, safer, and more efficient technological solutions.

### 🚗 Abstract

- Develop Vehicle Detection Models Accurately
  
Implement computer vision algorithms based on geometric shapes, including bounding boxes and polygons, for the robust detection and localization of vehicles in a variety of settings and conditions.

- Improve feature recognition and segmentation
  
Make use of geometric annotations to enhance key vehicle feature extraction and analysis for accurate segmentation that is used in applications like damage assessment and vehicle classification.

- Allow Real-Time Tracking and Analysis
  
Design algorithms that can apply geometric shapes towards real-time vehicle movement tracking for autonomous driving and support for management systems in traffic.

- Annotated Vehicle Dataset
  
The data comprises fully annotated vehicle images through geometrical shapes bounding box polygons keypoints suitable for any applications: detection, segmentation tracking.

- Trained Computer Vision Models
  
Highly performing machine learning models trained to identify, segment, and analyze vehicles under a variety of scenarios with a focus on occlusions and low-visibility conditions.

### 🚗 Project Method

- Imports
```   
import numpy as np
import cv2
import threading
```
• numpy: Although unused in this code, it is often used in OpenCV for image manipulation (e.g., arrays for images).

• cv2: OpenCV's Python library for computer vision tasks such as image processing, object detection, and video manipulation.

• threading: Enables concurrent processing to handle multiple video files simultaneously, optimizing runtime when processing multiple videos.

---

- process_video Function
```   
car_cascade = cv2.CascadeClassifier(haar_cascade_path)
```
• Haar cascades are pre-trained object detection classifiers.

• Here, a Haar cascade trained to detect cars is loaded.

```
cap = cv2.VideoCapture(video_path)
```
• Opens a video file for frame-by-frame processing.
```
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
```
• Retrieves video properties such as width, height, and frame rate (FPS) for output video creation.
```
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
```
• Sets up the codec (mp4v) for saving the processed video and creates a VideoWriter object.
```
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cars = car_cascade.detectMultiScale(gray, 1.3, 5)
```
• Read each frame (cap.read()).

• Converts the frame to grayscale (cv2.cvtColor) as Haar cascades work best on single-channel images.

• Detects cars using car_cascade.detectMultiScale():

    • 1.3: Scale factor (how much the image size is reduced at each scale).
    
    • 5: Minimum neighbors (the number of rectangles around a detected region for it to be considered valid).
```
cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
```
• Draws rectangles around detected cars on the frame.
• (0, 0, 255) specifies the rectangle color (red), and 3 is the thickness.
```
out.write(frame)
```
• Writes the processed frame to the output video.
```
cap.release()
out.release()
```
• Frees up memory by releasing the video capture and writer objects.

---

- Paths
```   
haar_cascade = 'haarcascade_car.xml'
videos = ['Car Set 1.mp4', 'Car Set 2.mp4', ...]
output_videos = ['Car Detector 1.mp4', 'Car Detector 2.mp4', ...]
```
• haar_cascade: Path to the Haar cascade XML file for car detection.

• videos: List of input video paths.

• output_videos: Corresponding output paths for processed videos.

---

- Multi-threading
```   
threads = []
for i in range(len(videos)):
    thread = threading.Thread(target=process_video, args=(videos[i], output_videos[i], haar_cascade))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()
```
• Purpose: Allows simultaneous processing of multiple videos to save time.

• Steps:

    • A new thread is created for each video using threading. Thread with process_video as the target function.
  
    • Threads are started using thread.start().
  
    • thread.join() ensures the main program waits for all threads to finish before proceeding.

---

- Completion Message
```   
print("Processing complete. All videos have been saved.")
```
• Indicates that all video processing is done.

---
Overall Workflow 
1. Open the video file and load the Haar Cascade model. 
2. Read frames from the video in a loop. 
3. Convert each frame to grayscale and detect cars using the Haar Cascade. 
4. Draw bounding boxes around detected cars. 
5. Save the processed frames to a new video file. 
6. Release resources when processing is complete.

### 🚗 Conclusion

- This program is used to detect cars in a video using a Haar Cascade classifier as the feature descriptor. It takes frames from a given video file, detects cars, draws rectangles around them, and then writes the video output. In testing the system on five videos, it was observed that for some of the videos the system was only able to identify one car per frame, while for other videos the system was able to identify multiple cars in the same frame. This variability may be attributed to the quality of the videos used, lighting, camera view point, or just the nature of the Haar Cascade classifier. There could also be false negatives and inconsistent detections due to the use of pre-trained Haar Cascade features in the system (haarcascade_car.xml). Despite the fact that the code does its job of detecting faces in videos successfully, there are built-in object detection models like YOLO and SSD that could be used, or detection parameters could be tweaked for better results, or the quality of the videos could be enhanced using some pre-processing techniques. In summary, the system presented herein exhibits functional performance in car detection, but there is certainly room for improvement in terms of accuracy and stability.

### 🚗 Additional Materials

[] Drawing Rectangles On Cars On The Highway

```ruby
import numpy as np
# (Not used in this script, but often included for array operations in OpenCV)
import cv2
# OpenCV library for image and video processing
import threading
# Used for concurrent execution of video processing tasks

# Function to process a single video file
def process_video(video_path, output_path, haar_cascade_path):
    # Load the Haar Cascade for car detection from the specified XML file
    car_cascade = cv2.CascadeClassifier(haar_cascade_path)

    # Open the input video file
    cap = cv2.VideoCapture(video_path)

    # Retrieve video properties such as width, height, and frames per second (fps)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec for the output video file and initialize VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Loop through each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()
        # Read a frame from the video
        if not ret:
        # Break the loop if no more frames are available
            break

        # Convert the frame to grayscale (required for Haar cascade detection)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect cars in the frame using the Haar cascade
        cars = car_cascade.detectMultiScale(gray, 1.3, 5)

        # Draw rectangles around detected cars
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            # Red rectangle, thickness 3

        # Write the processed frame with annotations to the output video
        out.write(frame)

    # Release resources to free up memory
    cap.release()
    # Close the video file
    out.release()
    # Close the output file

# Path to the Haar cascade XML file for car detection
haar_cascade = 'haarcascade_car.xml'
# List of input video file paths
videos = [
    'Car Set 1.mp4',
    'Car Set 2.mp4',
    'Car Set 3.mp4',
    'Car Set 4.mp4',
    'Car Set 5.mp4'
]
# List of corresponding output video file paths
output_videos = [
    'Car Detector 1.mp4',
    'Car Detector 2.mp4',
    'Car Detector 3.mp4',
    'Car Detector 4.mp4',
    'Car Detector 5.mp4'
]

# List to hold thread objects
threads = []

# Create and start a thread for processing each video
for i in range(len(videos)):
# Initialize a new thread to process the video using the `process_video` function
    thread = threading.Thread(target=process_video, args=(videos[i], output_videos[i], haar_cascade))
    threads.append(thread)
    # Add the thread to the list
    thread.start()
    # Start the thread

# Wait for all threads to finish processing
for thread in threads:
    thread.join()
    # Block the main thread until the current thread completes

# Print a message when all videos have been processed
print("Processing complete. All videos have been saved.")
```  
---

![image](https://github.com/user-attachments/assets/bac81746-5142-4d18-99b3-fbec35d0f53d)

![image](https://github.com/user-attachments/assets/2ee39d3d-978a-411f-b2bc-e4eef5485554)

![image](https://github.com/user-attachments/assets/be388145-0dbb-4a5d-ab50-06e262c4e431)

![image](https://github.com/user-attachments/assets/b84a2b70-0291-4999-8847-81a413afbc77)

![image](https://github.com/user-attachments/assets/92328f6a-f9ff-49dd-bae1-fe9bc259d0ec)

---

### References

https://github.com/misbah4064/car_detector_haarcascades

https://www.youtube.com/watch?v=zc6AP7B-CgI







