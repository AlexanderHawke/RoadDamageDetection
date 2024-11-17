import os
import json
from sklearn.model_selection import KFold
from ultralytics import YOLO

# Load your dataset paths from config.json
with open("config.json", "r") as f:
    config = json.load(f)

base_path = config["base_path"]

all_train_images = []
all_train_labels = []

# Collect all training images and labels
for train_dir in config["train_dirs"]:
    images_dir = os.path.join(base_path, train_dir, "images")
    annotations_dir = os.path.join(base_path, train_dir, "annotations", "xmls")

    # List all image files in the images directory
    images = [os.path.join(images_dir, img) for img in os.listdir(images_dir) if img.endswith(".jpg")]
    
    # Corresponding label paths based on image names
    labels = [os.path.join(annotations_dir, img.replace(".jpg", ".txt")) for img in os.listdir(images_dir) if img.endswith(".jpg")]

    # Ensure that the annotation files exist for each image
    valid_images = []
    valid_labels = []
    for img, lbl in zip(images, labels):
        if os.path.exists(lbl):  # Only include images with corresponding .txt labels
            valid_images.append(img)
            valid_labels.append(lbl)

    all_train_images.extend(valid_images)
    all_train_labels.extend(valid_labels)

# Parameters for cross-validation
k = 5  # Number of folds
epochs = 50
img_size = 640
results = []

kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Perform k-fold cross-validation
for fold, (train_index, val_index) in enumerate(kf.split(all_train_images)):
    print(f"Starting fold {fold + 1} of {k}...")

    # Get train and validation splits for this fold
    train_images = [all_train_images[i] for i in train_index]
    val_images = [all_train_images[i] for i in val_index]
    train_labels = [all_train_labels[i] for i in train_index]
    val_labels = [all_train_labels[i] for i in val_index]

    # Create temporary directories and YAML configuration for this fold
    train_dir = f"temp_data/fold_{fold}/train/"
    val_dir = f"temp_data/fold_{fold}/val/"
    os.makedirs(train_dir + "images", exist_ok=True)
    os.makedirs(train_dir + "labels", exist_ok=True)
    os.makedirs(val_dir + "images", exist_ok=True)
    os.makedirs(val_dir + "labels", exist_ok=True)

    for img, label in zip(train_images, train_labels):
        os.symlink(img, os.path.join(train_dir, "images", os.path.basename(img)))
        os.symlink(label, os.path.join(train_dir, "labels", os.path.basename(label)))

    for img, label in zip(val_images, val_labels):
        os.symlink(img, os.path.join(val_dir, "images", os.path.basename(img)))
        os.symlink(label, os.path.join(val_dir, "labels", os.path.basename(label)))

    # Write a temporary YAML config file for this fold
    yaml_content = f"""
    train: {train_dir}images
    val: {val_dir}images
    nc: 4
    names: ['D00', 'D10', 'D20', 'D40']
    """
    yaml_file = f"temp_data/fold_{fold}/road_damage.yaml"
    with open(yaml_file, "w") as f:
        f.write(yaml_content)

    # Train YOLOv8 on this fold
    model = YOLO("yolov8n.yaml")
    fold_results = model.train(data=yaml_file, epochs=epochs, imgsz=img_size)
    results.append(fold_results)

    # Clean up (optional, remove temp directories if desired)
    # shutil.rmtree(train_dir)
    # shutil.rmtree(val_dir)

# After all folds, calculate average performance metrics
avg_precision = sum(r['metrics/precision'] for r in results) / k
avg_recall = sum(r['metrics/recall'] for r in results) / k
avg_map = sum(r['metrics/mAP_0.5'] for r in results) / k

print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average mAP@0.5: {avg_map:.4f}")
