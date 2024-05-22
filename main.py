import torch

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

# Images
img = "https://static01.nyt.com/images/2020/12/11/business/11wheels/10wheels-videoSixteenByNine3000.jpg?year=2020&h=1688&w=3000&s=f13a7c2427cb2ef5ba3b83e0ef7de451ca34b7b33046eb2f2423242e435127c3&k=ZQJBKqZ0VN&tw=1"  # or file, Path, PIL, OpenCV, numpy, list

model.conf = 0.50

# Inference
results = model(img)

# Results
print(results.pandas().xyxy[0])
