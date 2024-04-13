import cv2


def draw_bounding_boxes(image_path, bbox_file_path, output_path):
    # Read the image
    image = cv2.imread(image_path)

    # Read bounding box information from the YOLO format file
    with open(bbox_file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        # Parse YOLO format: class x_center y_center width height
        class_id, x_center, y_center, width, height = map(float, line.strip().split())

        # Convert YOLO coordinates to image coordinates
        img_height, img_width, _ = image.shape
        x_center = int(x_center * img_width)
        y_center = int(y_center * img_height)
        box_width = int(width * img_width)
        box_height = int(height * img_height)

        # Calculate box coordinates
        x_min = max(0, int(x_center - box_width / 2))
        y_min = max(0, int(y_center - box_height / 2))
        x_max = min(img_width, int(x_center + box_width / 2))
        y_max = min(img_height, int(y_center + box_height / 2))

        # Draw bounding box on the image
        color = (0, 255, 0)  # Green color for the bounding box
        thickness = 2
        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

    # Save or display the image with bounding boxes
    # cv2.imwrite(output_path, image)
    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
image_path = '/home/masud.rana/Documents/Learning_Project/Important/XAI/data/yolo_data/train/images/181.jpg'
bbox_file_path = '/home/masud.rana/Documents/Learning_Project/Important/XAI/data/yolo_data/train/labels/181.txt'
output_path = ''

draw_bounding_boxes(image_path, bbox_file_path, output_path)
