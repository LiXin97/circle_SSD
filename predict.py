import cv2.cv2 as cv
import argparse

from src.net import *
from src.data import MyDataset
from src.utils.utils import *
from torch.nn import functional as F

device = try_gpu()
# model_path = 'model/0.pth'
# img_path = 'data/circle_train/images/0.png'

def offset_inverse(anchors, offset_preds):
    """Predict bounding boxes based on anchor boxes with predicted offsets."""
    anc = circle_wh_to_radius(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.concat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = circle_radius_to_wh(pred_bbox)
    return predicted_bbox

def nms(boxes, scores, iou_threshold):
    """Sort confidence scores of predicted bounding boxes."""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # Indices of predicted bounding boxes that will be kept
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = circle_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)

def multicircle_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """Predict bounding boxes using non-maximum suppression."""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # Find all non-`keep` indices and set the class to background
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # Here `pos_threshold` is a threshold for positive (non-background)
        # predictions
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat(
            (class_id.unsqueeze(1), conf.unsqueeze(1), predicted_bb), dim=1)
        out.append(pred_info)
    return torch.stack(out)

def predit(model_path, img_path):
    net = torch.load( model_path )
    X = torchvision.io.read_image(img_path).unsqueeze(0).float()
    net.eval()

    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = multicircle_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    predit_output = output[0, idx]

    threshold = 0.1
    rotateAngle = 0
    startAngle = 0
    endAngle = 360
    thickness = 3
    lineType = 1
    colors = [[0,0,255], [0,255,0], [255,0,0]]

    img = np.ones((256, 256, 3), np.uint8) * 255
    for row in predit_output:
        score = float(row[1])
        if score < threshold:
            continue

        ptCenter = (int(row[2]*256), int(row[3]*256))
        axesSize = (int(row[4]*256), int(row[5]*256))
        cv.ellipse(img, ptCenter, axesSize, rotateAngle, startAngle, endAngle, colors[int(row[0])], thickness, lineType)
    cv.imshow('img', img)
    cv.waitKey()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path",
                        help="path to saved model", type=str, required=True)
    parser.add_argument("-t", "--test_img_path",
                        help="path to test img", type=str, required=True)
    args = parser.parse_args()

    predit(args.model_path, args.test_img_path)