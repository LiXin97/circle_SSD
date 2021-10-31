import torch
import torchvision
import torch.nn as nn

radiuses = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(radiuses[0]) + len(ratios[0]) - 1

cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.SmoothL1Loss(reduction='none')

def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)

def bcircle_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)

def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)

def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk

def multicicle_prior(data, radiuses, ratios):
    """Generate anchor cicles with different radius on each pixel."""
    in_height, in_width = data.shape[-2:]
    device, num_radius, num_ratios = data.device, len(radiuses), len(ratios)
    cicles_per_pixel = (num_radius + num_ratios - 1)
    radiuses_tensor = torch.tensor(radiuses, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)
    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y axis
    steps_w = 1.0 / in_width  # Scaled steps in x axis

    # Generate all center points for the anchor boxes
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w)
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)


    # Generate `circles_per_pixel`
    # used to create anchor circle corner coordinates (xmin, xmax, ymin, ymax)
    radius = torch.cat((radiuses_tensor.reshape(-1, 1), radiuses_tensor[0].repeat(len(ratio_tensor) - 1).reshape(-1, 1)))
    ratio = torch.cat((ratio_tensor[0].repeat(len(radiuses_tensor) - 1).reshape(-1, 1), ratio_tensor.reshape(-1, 1)))

    radius_xy = torch.cat((radius, radius * ratio), dim=1)
    # Divide by 2 to get half height and half width
    anchor_manipulations = radius_xy.repeat(in_height * in_width, 1)

    # Each center point will have `cicles_per_pixel` number of anchor circles, so
    # generate a grid of all anchor circle centers with `cicles_per_pixel` repeats
    out_grid = torch.stack([shift_x, shift_y],
                           dim=1).repeat_interleave(cicles_per_pixel, dim=0)
    output = torch.cat( (out_grid, anchor_manipulations), 1 )
    return output.unsqueeze(0)


def blk_forward(X, blk, radiuses, ratios, cls_predictor, bcircle_predictor):
    Y = blk(X)
    anchors = multicicle_prior(Y, radiuses, ratios)
    cls_preds = cls_predictor(Y)
    bcicle_preds = bcircle_predictor(Y)
    return (Y, anchors, cls_preds, bcicle_preds)

class TinyCircleSSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinyCircleSSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # 即赋值语句 `self.blk_i = get_blk(i)`
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bcircle_{i}', bcircle_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bcircle_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # `getattr(self, 'blk_%d' % i)` 即访问 `self.blk_i`
            X, anchors[i], cls_preds[i], bcircle_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), radiuses[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bcircle_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bcircle_preds = concat_preds(bcircle_preds)
        return anchors, cls_preds, bcircle_preds





def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox

def cls_eval(cls_preds, cls_labels):
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())

def mask_iou(target_mask, prediction_mask):
    intersection = torch.logical_and(target_mask, prediction_mask)
    union = torch.logical_or(target_mask, prediction_mask)
    return torch.sum(intersection, dim=[1,2]) / torch.sum(union,dim =[1,2])

def mask_circle(circles):
    masks = torch.zeros((circles.shape[0], 256, 256), device=circles.device, dtype = bool)
    pixel_circles = circles * 256


    return True

def circle2box(circles):
    boxes = circles.clone()
    boxes[:, 0] = circles[:, 0] - circles[:, 2]
    boxes[:, 1] = circles[:, 1] - circles[:, 3]
    boxes[:, 2] = circles[:, 0] + circles[:, 2]
    boxes[:, 3] = circles[:, 1] + circles[:, 3]
    return boxes

def circle_iou(circles1, circles2):
    boxes1 = circle2box(circles1)
    boxes2 = circle2box(circles2)

    """Compute pairwise IoU across two lists of anchor or bounding boxes."""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas

def assign_anchor_to_bcircle(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes."""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Element x_ij in the i-th row and j-th column is the IoU of the anchor
    # box i and the ground-truth bounding box j
    jaccard = circle_iou(anchors, ground_truth)
    # Initialize the tensor to hold the assigned ground-truth bounding box for
    # each anchor
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # Assign ground-truth bounding boxes according to the threshold
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)  # Find the largest IoU
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map

def circle_radius_to_wh(circles):
    """Convert from (center, radius_x, radius_y) to (center, width, height)."""
    x, y, radius_x, rasius_y = circles[:, 0], circles[:, 1], circles[:, 2], circles[:, 3]
    w = radius_x * 2.
    h = rasius_y * 2.
    circles = torch.stack((x, y, w, h), axis=-1)
    return circles

def circle_wh_to_radius(circles):
    """Convert from (center, width, height) to (center, radius_x, radius_y)."""
    x, y, w, h = circles[:, 0], circles[:, 1], circles[:, 2], circles[:, 3]
    radius_x = w / 2.
    rasius_y = h / 2.
    circles = torch.stack((x, y, radius_x, rasius_y), axis=-1)
    return circles

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """Transform for anchor box offsets."""
    c_anc = circle_radius_to_wh(anchors)
    c_assigned_bb = circle_radius_to_wh(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset

def multicircle_target(anchors, labels):
    """Label anchor boxes using ground-truth bounding boxes."""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bcircle(label[:, 1:], anchors,
                                                 device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # Initialize class labels and assigned bounding box coordinates with
        # zeros
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # Label classes of anchor boxes using their assigned ground-truth
        # bounding boxes. If an anchor box is not assigned any, we label its
        # class as background (the value remains zero)
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # Offset transformation
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)