from ultralytics import YOLO

# Initialize the YOLO model with a valid identifier
model = YOLO("./models/best.pt")

# Perform prediction on the input video
results = model.predict(source="./input-videos/08fd33_4.mp4", save=True)

# Print the results of the first frame
print(results[0])

print("--------------------------------")

# Print the bounding boxes in the first frame
for box in results[0].boxes:
    print(box)

