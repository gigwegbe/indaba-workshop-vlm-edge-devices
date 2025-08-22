import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
# -----------------------------
# Load model once
# -----------------------------
detect_fn = tf.saved_model.load("ei-tensorflow-savedmodel-model/")

# -----------------------------
# Load labels once
# -----------------------------
labels = {}
with open("map.txt", "r") as f:
    for i, line in enumerate(f.readlines()):
        labels[i] = line.strip()

# -----------------------------
# Colors for bounding boxes
# -----------------------------
colors = [
    (0, 255, 0),     # Green
    (0, 0, 255),     # Red
    (255, 0, 0),     # Blue
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
    (128, 0, 128),   # Purple
    (0, 128, 128),   # Teal
    (128, 128, 0)    # Olive
]

def detect_objects(image_path: str, output_dir: str = "./detected", threshold: float = 0.45) -> str:
    """
    Run object detection on an image and save annotated result.

    Args:
        image_path (str): Path to input image.
        output_dir (str): Directory to save output image.
        threshold (float): Confidence threshold for filtering detections.

    Returns:
        str: Path to saved annotated image.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    height, width, _ = image.shape
    input_size = (640, 640)
    resized_image = cv2.resize(image, input_size)

    # Normalize to [0,1] and add batch dimension
    input_tensor = tf.convert_to_tensor(resized_image, dtype=tf.float32)
    input_tensor = tf.expand_dims(input_tensor, axis=0) / 255.0  

    # Run inference
    detections = detect_fn(input_tensor)
    preds = detections[0].numpy()[0]  # (num_boxes, 11)

    # Split predictions
    boxes = preds[:, 0:4]
    objectness = preds[:, 4]
    class_scores = preds[:, 5:]

    # Get class and confidence
    classes = np.argmax(class_scores, axis=-1)
    scores = objectness * np.max(class_scores, axis=-1)

    # Apply threshold
    mask = scores > threshold
    boxes, scores, classes = boxes[mask], scores[mask], classes[mask]

    # Draw detections
    for (x, y, w, h), score, cls in zip(boxes, scores, classes):
        x1 = int((x - w/2) * width)
        y1 = int((y - h/2) * height)
        x2 = int((x + w/2) * width)
        y2 = int((y + h/2) * height)

        color = colors[cls % len(colors)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        class_name = labels.get(cls, str(cls))
        label = f"{class_name}: {score:.2f}"

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (x1, y1 - th - baseline), (x1 + tw, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_detections.jpg")
    cv2.imwrite(output_path, image)
    print(f"Saved annotated image to {output_path}")
    return output_path



def load_image(image_path: str):
    """
    Load an image from disk and return as a PIL Image in RGB format.

    Args:
        image_path (str): Path to image.

    Returns:
        PIL.Image.Image: Image in RGB mode.
    """
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    return image
