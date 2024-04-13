import os

import shap
import cv2
from skimage.segmentation import mark_boundaries
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import ops
from ultralytics.data.augment import LetterBox
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import torch
import torchvision.ops

output_path = '/home/masud.rana/Documents/Learning_Project/Important/XAI/data/output/shap'
input_path = '/home/masud.rana/Documents/Learning_Project/Important/XAI/data/error_images'

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

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(device)

model, ckpt = attempt_load_one_weight('yolo_v8s_640.pt')
model.float()
model = model.to(device)
model.eval()
print()


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f'input shape last dimension expected 4 but input shape is {x.shape}'
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    dw = x[..., 2] / 2  # half-width
    dh = x[..., 3] / 2  # half-height
    y[..., 0] = x[..., 0] - dw  # top left x
    y[..., 1] = x[..., 1] - dh  # top left y
    y[..., 2] = x[..., 0] + dw  # bottom right x
    y[..., 3] = x[..., 1] + dh  # bottom right y
    return y


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks + num_classes) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ..., class_prob1, class_prob2, ...).
    """

    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + nm + nc), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence

        if not x.shape[0]:
            continue

        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i], cls[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask, cls), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        output[xi] = x[i]
        if (time.time() - t) > max_time_img:
            print('Time limit exceeded')
            break  # time limit exceeded

    return output


def predict_torch_batch(images):
    # print("start_predict")
    # print(images.shape)
    # print("okay start")

    # Ensure the model is in evaluation mode
    # model.eval()

    # List to store predictions for each image in the batch
    batch_preds = []

    for image in images:
        # print("First image")
        # image = images[i]
        # print(image.shape)

        convert_tensor = transforms.ToTensor()

        # converting image to tensor
        image_tensor = convert_tensor(image)
        # adding dimension - [B,C,W,H]
        image_tensor = image_tensor.unsqueeze(0)
        # converting tensor to float (Float32)
        image_tensor = image_tensor.float()
        image_tensor = image_tensor.to(device)

        # print(image_tensor.shape)

        with torch.no_grad():
            results = model(image_tensor)[0]

        conf = 0.25
        iou = 0.7
        agnostic_nms = False
        max_det = 300
        classes = None

        preds = non_max_suppression(results,
                                    conf,
                                    iou,
                                    agnostic=agnostic_nms,
                                    max_det=max_det,
                                    classes=classes,
                                    nc=0)

        if not preds or not preds[0].shape[0]:
            pred = np.zeros(15).reshape(1, -1)
            # print(pred)
            batch_preds.append(pred)
        else:

            skipped_values = preds[0][:, 6:]
            # print(skipped_values)

            # Find the maximum value along each column
            max_values = torch.max(skipped_values, dim=0).values
            max_values = max_values.cpu() if max_values.is_cuda else max_values

            # Convert the result to a NumPy array if needed
            pred = max_values.numpy().reshape(1, -1)

            # print(pred)

            batch_preds.append(pred)

    # print("End predict")

    return np.concatenate(batch_preds, axis=0)


def generate_shap_values(image_path, save_file, true_labels=None):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to match the input shape
    image = cv2.resize(image, (640, 640))
    images = image.reshape(1, *image.shape)

    topk = 5
    batch_size = 1
    n_evals = 1000

    masker_blur = shap.maskers.Image("blur(128,128)", (640, 640, 3))

    explainer = shap.Explainer(predict_torch_batch, masker_blur, output_names=selected_classes)
    shap_values = explainer(
        images,
        max_evals=n_evals,
        batch_size=batch_size,
        outputs=shap.Explanation.argsort.flip[:topk],
    )

    shap_values.data = shap_values.data.reshape(640, 640, 3)
    shap_values.values = [val for val in np.moveaxis(shap_values.values[0], -1, 0)]

    shap.image_plot(
        shap_values=shap_values.values,
        pixel_values=shap_values.data,
        labels=shap_values.output_names,
        true_labels=[true_labels],
        show=False
    )

    plt.savefig(save_file)


def load_labels(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    return [line.strip().split(' ') for line in lines]


def get_labels(label_file):
    labels = load_labels(label_file)
    t_labels = [int(l[0]) for l in labels]
    cls_labels = [selected_classes[l] for l in t_labels]
    return cls_labels


def main():
    processing = 1
    for class_name in os.listdir(input_path):
        class_path = os.path.join(input_path, class_name)
        save_path = os.path.join(output_path, class_name)
        os.makedirs(save_path, exist_ok=True)

        for image_file in os.listdir(class_path):
            if image_file.endswith('txt'):
                continue
            print(f"Working for {processing}: {image_file}")
            image_path = os.path.join(class_path, image_file)

            label_file = os.path.join(class_path, image_file.replace('.jpg', '.txt'))
            true_labels = get_labels(label_file)
            save_file = os.path.join(save_path, image_file)

            generate_shap_values(image_path, save_file, true_labels)

            print(f"Done for {processing}:{image_file}")
            processing += 1


if __name__ == '__main__':
    main()
