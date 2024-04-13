import os
import shutil
from collections import defaultdict

import cv2
from ultralytics import YOLO

model = YOLO('yolo_v8s_640.pt')

data_path = '/home/masud.rana/Documents/Learning_Project/Important/XAI/data/yolo_data/val'
image_dir = os.path.join(data_path, 'images')
label_dir = os.path.join(data_path, 'labels')
output_dir = '/home/masud.rana/Documents/Learning_Project/Important/XAI/data/error_images'

error_image_map = {i: 0 for i in range(15)}
MAX_IMAGES = 6


def load_labels(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    return [line.strip().split(' ') for line in lines]


def get_labels(label_file):
    labels = load_labels(label_file)
    return [int(l[0]) for l in labels]


def get_error_labels(true_label, predicted_label):
    error_label = []
    for t in true_label:
        if t not in predicted_label:
            error_label.append(t)
    return error_label


def save_error_image(error_labels, image_file, label_file, names):
    image_path = os.path.join(image_dir, image_file)
    for el in error_labels:
        if error_image_map.get(el) < MAX_IMAGES:
            os.makedirs(os.path.join(output_dir, names[el]), exist_ok=True)
            shutil.copy(image_path, os.path.join(output_dir, names[el], image_file))
            shutil.copy(label_file, os.path.join(output_dir, names[el], image_file.replace('.jpg', '.txt')))
            print(f"Image saved for {names[el]}")
            error_image_map[el] += 1
            return


def should_continue():
    for k, v in error_image_map.items():
        if v < MAX_IMAGES:
            return True
    print("All done...")
    return False


def main():
    # Iterate through images
    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)

        # Load corresponding label file
        label_file = os.path.join(label_dir, image_file.replace('.jpg', '.txt'))
        true_labels = get_labels(label_file)

        # Perform YOLO-v8 prediction
        predictions = model(image_path)

        predicted_labels = predictions[0].boxes.cls.cpu().numpy()
        names = predictions[0].names

        error_labels = get_error_labels(true_labels, predicted_labels)

        if not error_labels:
            continue

        save_error_image(error_labels, image_file, label_file, names)

        if not should_continue():
            break

        # Compare predictions with actual labels
        # for prediction in predictions:
        #     # Check if the prediction matches any actual label
        #     if prediction not in labels:
        #         # Save the image with incorrect prediction
        #         cv2.imwrite(f'incorrect_predictions/{image_file}', image)
        #
        #         # Save the incorrect label information
        #         with open(f'incorrect_predictions/{image_file.replace(".jpg", ".txt")}', 'w') as f:
        #             f.write('\n'.join(labels))
        #         break


if __name__ == "__main__":
    main()
