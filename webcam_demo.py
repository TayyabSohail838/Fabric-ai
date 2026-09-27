import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from PIL import Image

class_names = ['defect_free', 'hole', 'horizontal', 'lines', 'stain', 'verticle']

# Same architecture as build_model() in app.py: MobileNetV3-small with a new head.
print("Loading model...")
model = mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
model.load_state_dict(torch.load("fabric_model.pt", map_location="cpu"))
model.eval()
print("Model loaded.")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        return class_names[pred.item()], conf.item() * 100

print("Opening webcam...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Could not open webcam. Close other apps using the camera (Zoom, Teams, browser tabs) and try again.")
    exit()

print("Webcam opened successfully. Starting preview window...")

result_text = ""
result_color = (255, 255, 255)
capture_requested = False
btn_radius = 25

def mouse_callback(event, x, y, flags, param):
    global capture_requested
    if event == cv2.EVENT_LBUTTONDOWN:
        btn_x, btn_y = param
        dist = ((x - btn_x) ** 2 + (y - btn_y) ** 2) ** 0.5
        if dist <= btn_radius + 10:
            capture_requested = True

cv2.namedWindow("Fabric Defect Detection")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to read frame from webcam.")
        break

    h, w = frame.shape[:2]
    display = frame.copy()
    btn_center = (w // 2, h - 50)

    cv2.setMouseCallback("Fabric Defect Detection", mouse_callback, btn_center)

    cv2.putText(display, "Click the green button or press SPACE | Q to quit",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.circle(display, btn_center, btn_radius, (255, 255, 255), 3)
    cv2.circle(display, btn_center, btn_radius - 7, (0, 200, 0), -1)

    if result_text:
        cv2.putText(display, result_text, (20, h - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, result_color, 2)

    cv2.imshow("Fabric Defect Detection", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if key == ord(' ') or capture_requested:
        capture_requested = False
        label, confidence = predict(frame)
        result_text = f"{label} ({confidence:.1f}%)"
        result_color = (0, 255, 0) if label == "defect_free" else (0, 0, 255)
        print(f"Captured -> {result_text}")

cap.release()
cv2.destroyAllWindows()