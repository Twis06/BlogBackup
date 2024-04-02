---
title: "Installing Yolov8 on apple silicon m3 pro (1)"
datePublished: Tue Apr 02 2024 11:50:41 GMT+0000 (Coordinated Universal Time)
cuid: cluibie7h000508jrhzn4elqa
slug: installing-yolov8-on-apple-silicon-m3-pro-1
cover: https://cdn.hashnode.com/res/hashnode/image/stock/unsplash/ieic5Tq8YMk/upload/71544e33f457aa7fe1f3d0783919f628.jpeg
tags: developer

---

[  
](https://ultralytics.com/)Ultralytics YOLOv8 is a cutting-edge, state-of-the-art (SOTA) model that builds upon the success of previous YOLO versions and introduces new features and improvements to further boost performance and flexibility. YOLOv8 is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide range of object detection and tracking, instance segmentation, image classification and pose estimation tasks.

## Installing requirements

Reaching version 8, all the requirements required by yolo is compresses in a single package, ultralytics. The package can be directly downloaded from pip with the commands

```powershell
pip install ultralytics
```

if you haven't had pip installed, follow the instructions.

* Open up terminal
    
    Open "spotlight search" and search for terminal.
    

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1712043287977/9a065a6a-a0ec-4d15-94e3-90f778366920.png align="center")

* Check your python version
    

```objectivec
python3 --version 
```

* Run the codes and install pip
    

```objectivec
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
```

## Download model

Once ultralytics is installed, we can import yolo and download the v8 model.

Here is the example code,

```python
import cv2
from ultralytics import YOLO
import numpy as np
import torch


cap = cv2.VideoCapture("app.mp4") # Video source

model = YOLO("yolov8m.pt") # Model selection

while True:
    ret, frame = cap.read()
    if not ret:
        break  

    results = model(frame, device="mps") #Use this code when you want to utilize GPU
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")

    for cls, bbox in zip(classes, bboxes):
        (x,y,x2,y2) = bbox

        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
        cv2.putText(frame, str(cls), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,225), 2)

    cv2.imshow("Img", frame)
    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

*ret* is a boolean variable that returns true if the frame is available.

*frame* is an image array vector captured based on the default frames per second defined explicitly or implicitly.

Notably, you can choose camera input in this line.

```python
cap = cv2.VideoCapture("app.mp4")
```

"$" refers to the file under the same catalog

0, 1, 2 refers to different camera input.

## Labeling and tracking the objects

```python
import cv2
import numpy as np
from ultralytics import YOLO

from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors

from collections import defaultdict

track_history = defaultdict(lambda: [])
model = YOLO("yolov8m.pt")
names = model.model.names

# video_path = "/path/to/video/file.mp4"
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

result = cv2.VideoWriter("object_tracking.avi",
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (w, h))

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.track(frame, persist=True, verbose=False)
        boxes = results[0].boxes.xyxy.cpu()

        if results[0].boxes.id is not None:

            # Extract prediction results
            clss = results[0].boxes.cls.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.float().cpu().tolist()

            # Annotator Init
            annotator = Annotator(frame, line_width=2)

            for box, cls, track_id in zip(boxes, clss, track_ids):
                annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

                # Store tracking history
                track = track_history[track_id]
                track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
                if len(track) > 30:
                    track.pop(0)

                # Plot tracks
                print(track[-1])
                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                cv2.circle(frame, (track[-1]), 7, colors(int(cls), True), -1)
                cv2.polylines(frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2)

        result.write(frame)
        cv2.imshow("Img", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

result.release()
cap.release()
cv2.destroyAllWindows()
```

Here is the example code for tracking object.