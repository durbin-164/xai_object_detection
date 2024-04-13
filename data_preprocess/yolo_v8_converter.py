import json
import os.path
import shutil

import cv2

dataset_root = '/home/masud.rana/Documents/Learning_Project/Important/XAI/data/tt100k_2021'
train_data_path = os.path.join(dataset_root, 'train')
test_data_path = os.path.join(dataset_root, 'test')
annotation_path = os.path.join(dataset_root, 'annotations_all.json')
train_ids_path = os.path.join(train_data_path, 'ids.txt')
test_ids_path = os.path.join(test_data_path, 'ids.txt')

annotations = json.loads(open(annotation_path).read())
ids_train = open(train_ids_path).read().splitlines()
ids_test = open(test_ids_path).read().splitlines()

selected_classes = [
    'i5',
    'ip',
    'p5',
    'p23',
    'pne',
    'pl40',
    'pl50',
    'pl80',
    'pl60',
    'pl100',
    'pl30',
    'pl5',
    'pn',
    'p11',
    'p12'
]

classes_mapper = {
    'i5': 0,
    'ip': 1,
    'p5': 2,
    'p23': 3,
    'pne': 4,
    'pl40': 5,
    'pl50': 6,
    'pl80': 7,
    'pl60': 8,
    'pl100': 9,
    'pl30': 10,
    'pl5': 11,
    'pn': 12,
    'p11': 13,
    'p12': 14
}

yolo_data_root = '/home/masud.rana/Documents/Learning_Project/Important/XAI/data/yolo_data'
yolo_train_path = os.path.join(yolo_data_root, 'train')
yolo_val_path = os.path.join(yolo_data_root, 'val')


def convert_box_in_yolo_format(label, shape, box):
    height, width, channels = shape
    xmin = box['xmin']
    xmax = box['xmax']
    ymin = box['ymin']
    ymax = box['ymax']

    b_h = (ymax - ymin) / float(height)
    b_w = (xmax - xmin) / float(width)

    x_c = ((xmin + xmax) / 2.0) / float(width)
    y_c = ((ymin + ymax) / 2.0) / float(height)

    label_int = classes_mapper[label]

    return label_int, x_c, y_c, b_w, b_h


def copy_image(image_loc, save_path, image_id):
    image_save_loc = os.path.join(save_path, 'images', f'{image_id}.jpg')
    shutil.copy2(image_loc, image_save_loc)


def write_labels(label_data, save_path, image_id):
    label_loc = os.path.join(save_path, 'labels', f'{image_id}.txt')
    with open(label_loc, 'w') as file:
        for data in label_data:
            file.write(f'{data[0]} {data[1]} {data[2]} {data[3]} {data[4]}\n')


def convert_to_yolo_format(annots, ids, image_path, save_path):
    missing = 0
    for i in ids:
        if not annots['imgs'].get(i):
            missing += 1
            continue

        objs = annots['imgs'][i]['objects']

        if not objs:
            missing += 1
            continue

        image_loc = os.path.join(image_path, f'{i}.jpg')
        image = cv2.imread(image_loc)
        image_shape = image.shape

        labels_data = []

        for obj in objs:
            label = obj['category']
            box = obj['bbox']

            if label not in selected_classes:
                continue

            labels_data.append(convert_box_in_yolo_format(label, image_shape, box))

        if not labels_data:
            missing += 1
            continue

        copy_image(image_loc, save_path, i)
        write_labels(labels_data, save_path, i)

    print(f"Total images: {len(ids)}")
    print(f"Missing images: {missing}")


if __name__ == '__main__':
    # Convert train data
    print("Training data .....")
    convert_to_yolo_format(annotations, ids_train, train_data_path, yolo_train_path)
    # Convert Test Data
    print("Testing data ......")
    convert_to_yolo_format(annotations, ids_test, test_data_path, yolo_val_path)
