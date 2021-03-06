import math
import torch
import numpy as np
import torch.nn as nn

if torch.cuda.is_available():
    bool_tensor = torch.cuda.BoolTensor
    float_tensor = torch.cuda.FloatTensor
else:
    bool_tensor = torch.BoolTensor
    float_tensor = torch.FloatTensor


class DBL(nn.Module):

    def __init__(self, in_channels, out_channels, k_size, stride=1, padding=True):

        super(DBL, self).__init__()

        pad = 0
        if padding:
            pad = (k_size - 1) // 2
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=k_size,
                              stride=stride,
                              bias=False,
                              padding=pad)

        self.bn = nn.BatchNorm2d(num_features=out_channels)

        self.l_relu = nn.LeakyReLU()

    def forward(self, input):

        output = self.conv(input)

        output = self.bn(output)

        output = self.l_relu(output)

        return output


class ResUnit(nn.Module):

    def __init__(self, in_channels, out_channels, k_size, stride=1, padding=True):

        super(ResUnit, self).__init__()

        self.conv_bn_relu1 = DBL(in_channels=in_channels,
                                 out_channels=out_channels // 2,
                                 k_size=1,
                                 stride=stride,
                                 padding=padding
                                 )

        self.conv_bn_relu2 = DBL(in_channels=out_channels // 2,
                                 out_channels=out_channels,
                                 k_size=k_size,
                                 stride=stride,
                                 padding=padding)

    def forward(self, input):

        shortcut = input

        output = self.conv_bn_relu1(input)

        output = self.conv_bn_relu2(output)

        output = output + shortcut

        return output


class ResN(nn.Module):

    def __init__(self, n_layer, in_channels, out_channels, k_size, stride=1, padding=True):

        super(ResN, self).__init__()
        self.res_layers = nn.ModuleList([ResUnit(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 k_size=k_size,
                                                 stride=stride,
                                                 padding=padding) for _ in range(n_layer)])

    def forward(self, input):

        output = input
        for layer in self.res_layers:
            output = layer(output)

        return output


class Darknet53(nn.Module):

    def __init__(self):

        super(Darknet53, self).__init__()

        self.conv1 = DBL(in_channels=3, out_channels=32,
                         k_size=3, stride=1, padding=True)
        self.conv2 = DBL(in_channels=32, out_channels=64,
                         k_size=3, stride=2, padding=True)

        self.res_1 = ResN(n_layer=1,
                          in_channels=64,
                          out_channels=64,
                          k_size=3,
                          stride=1,
                          padding=True)

        self.conv3 = DBL(in_channels=64, out_channels=128,
                         k_size=3, stride=2, padding=True)

        self.res_2 = ResN(n_layer=2,
                          in_channels=128,
                          out_channels=128,
                          k_size=3,
                          stride=1,
                          padding=True)

        self.conv4 = DBL(in_channels=128, out_channels=256,
                         k_size=3, stride=2, padding=True)

        self.res_8 = ResN(n_layer=8,
                          in_channels=256,
                          out_channels=256,
                          k_size=3,
                          stride=1,
                          padding=True)

        self.conv5 = DBL(in_channels=256, out_channels=512,
                         k_size=3, stride=2, padding=True)

        self.res_8_2 = ResN(n_layer=8,
                            in_channels=512,
                            out_channels=512,
                            k_size=3,
                            stride=1,
                            padding=True)

        self.conv6 = DBL(in_channels=512, out_channels=1024,
                         k_size=3, stride=2, padding=True)

        self.res_4 = ResN(n_layer=4,
                          in_channels=1024,
                          out_channels=1024,
                          k_size=3,
                          stride=1,
                          padding=True)

    def forward(self, input):

        output = self.conv1(input)
        output = self.conv2(output)

        output = self.res_1(output)

        output = self.conv3(output)

        output = self.res_2(output)

        output = self.conv4(output)

        output = self.res_8(output)

        route1 = output.clone()

        output = self.conv5(output)

        output = self.res_8_2(output)

        route2 = output.clone()

        output = self.conv6(output)

        output = self.res_4(output)

        route3 = output.clone()

        return route1, route2, route3


class YoloLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(YoloLayer, self).__init__()

        self.conv1 = DBL(in_channels=in_channels,
                         k_size=1,
                         out_channels=out_channels // 2,
                         stride=1,
                         padding=True)
        self.conv2 = DBL(in_channels=out_channels//2,
                         k_size=3,
                         out_channels=out_channels,
                         stride=1,
                         padding=True)
        self.conv3 = DBL(in_channels=out_channels,
                         k_size=1,
                         out_channels=out_channels // 2,
                         stride=1,
                         padding=True)
        self.conv4 = DBL(in_channels=out_channels//2,
                         k_size=3,
                         out_channels=out_channels,
                         stride=1,
                         padding=True)
        self.conv5 = DBL(in_channels=out_channels,
                         k_size=1,
                         out_channels=out_channels // 2,
                         stride=1,
                         padding=True)

        self.conv6 = DBL(in_channels=out_channels//2,
                         k_size=3,
                         out_channels=out_channels,
                         stride=1,
                         padding=True)

    def forward(self, input):

        net = self.conv1(input)
        net = self.conv2(net)
        net = self.conv3(net)
        net = self.conv4(net)

        net = self.conv5(net)

        route = net.clone()

        net = self.conv6(net)

        return route, net


class UpsampleLayer(nn.Module):

    def __init__(self, scale_factor, in_channels, out_channels, k_size, stride=1, padding=True):
        super(UpsampleLayer, self).__init__()
        self.conv = DBL(in_channels=in_channels, out_channels=out_channels,
                        k_size=k_size, stride=stride, padding=padding)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')

    def forward(self, input):

        output = self.conv(input)
        output = self.upsample(output)

        return output


class Yolov3(nn.Module):

    def __init__(self, n_classes):
        super(Yolov3, self).__init__()
        self.n_classes = n_classes
        self.darknet53 = Darknet53()

        self.yolo_layer1 = YoloLayer(in_channels=1024, out_channels=1024)
        self.conv1 = nn.Conv2d(
            in_channels=1024, out_channels=3 * (5 + self.n_classes), kernel_size=1, bias=True)
        self.upsample1 = UpsampleLayer(
            scale_factor=2, in_channels=512, out_channels=256, k_size=1, stride=1, padding=True)

        self.yolo_layer2 = YoloLayer(in_channels=768, out_channels=512)
        self.conv2 = nn.Conv2d(
            in_channels=512, out_channels=3 * (5 + self.n_classes), kernel_size=1, bias=True)
        self.upsample2 = UpsampleLayer(
            scale_factor=2, in_channels=256, out_channels=128, k_size=1, stride=1, padding=True)

        self.yolo_layer3 = YoloLayer(in_channels=384, out_channels=256)
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=3 * (5 + self.n_classes), kernel_size=1, bias=True)

    def forward(self, image):

        route1, route2, route3 = self.darknet53(image)

        inter_1, net1 = self.yolo_layer1(route3)

        feature_map1 = self.conv1(net1)

        inter_1 = self.upsample1(inter_1)

        concat1 = torch.cat([inter_1, route2], dim=1)

        inter_2, net2 = self.yolo_layer2(concat1)

        feature_map2 = self.conv2(net2)

        inter_2 = self.upsample2(inter_2)

        concat2 = torch.cat([inter_2, route1], dim=1)

        _, feature_map3 = self.yolo_layer3(concat2)

        feature_map3 = self.conv3(feature_map3)

        return feature_map1, feature_map2, feature_map3


def predict_transform(prediction, anchors, n_classes, image_size, device=None):

    batch_size = prediction.size(0)
    stride = image_size // prediction.size(2)

    grid_size = image_size // stride

    bbox_attrs = 5 + n_classes

    num_anchors = 3

    #prediction = [batch, 75, grid_size ,grid_size]

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)

    prediction = prediction.transpose(1,2).contiguous()

    prediction = prediction.view(batch_size, grid_size*grid_size, num_anchors, bbox_attrs)

    prediction = prediction.view(batch_size, grid_size, grid_size, num_anchors, bbox_attrs)

    box_centers, box_sizes, conf_logits, prob_logits = torch.split(
        prediction, [2, 2, 1, n_classes], dim=-1)

    rescaled_anchors = anchors / stride

    # Add the center offsets

    grid = np.arange(grid_size)

    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1).type(float_tensor)
    y_offset = torch.FloatTensor(b).view(-1,1).type(float_tensor)

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(grid_size, grid_size, num_anchors ,2).unsqueeze(0)

    # xy_offset = [13,13,1,2]

    box_centers = torch.sigmoid(box_centers)

    box_centers = box_centers + x_y_offset

    box_centers = box_centers * stride

    box_sizes = torch.exp(box_sizes) * rescaled_anchors

    box_sizes = box_sizes * stride

    boxes = torch.cat([box_centers, box_sizes], dim=-1)

    return x_y_offset, boxes, conf_logits, prob_logits, stride


def calculate_iou(pred_boxes, valid_true_boxes):

    # boxes = [grid_size, grid_size, 3, 2]
    # anchors = [3,2]
    pred_box_xy = pred_boxes[..., 0:2]
    pred_box_wh = pred_boxes[..., 2:4]

    # shape: [13, 13, 3, 1, 2]
    pred_box_xy = torch.unsqueeze(pred_box_xy, dim=-2)
    pred_box_wh = torch.unsqueeze(pred_box_wh, dim=-2)

    # [V, 2]
    true_box_xy = valid_true_boxes[:, 0:2]
    true_box_wh = valid_true_boxes[:, 2:4]

    # [13, 13, 3, 1, 2] & [V, 2] ==> [13, 13, 3, V, 2]
    intersect_mins = torch.max(pred_box_xy - pred_box_wh / 2.,
                               true_box_xy - true_box_wh / 2.)
    intersect_maxs = torch.min(pred_box_xy + pred_box_wh / 2.,
                               true_box_xy + true_box_wh / 2.)
    intersect_wh = torch.max(
        intersect_maxs - intersect_mins, torch.tensor([0.]).type(float_tensor))[0]

    # shape: [13, 13, 3, V]
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    # shape: [13, 13, 3, 1]
    pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
    # shape: [V]
    true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]
    # shape: [1, V]
    true_box_area = torch.unsqueeze(true_box_area, dim=0)

    # [13, 13, 3, V]
    iou = intersect_area / \
        (pred_box_area + true_box_area - intersect_area + 1e-10)

    return iou


def calculate_ignore_mask(pred_boxes, y_true, object_mask,  threshold):

    batch_size = pred_boxes.shape[0]

    ignore_mask = torch.ones(size=object_mask.shape).type(float_tensor)

    for idx in range(batch_size):

        valid_true_boxes = torch.masked_select(
            y_true[idx, ..., :4], object_mask[idx].type(bool_tensor))

        if valid_true_boxes.shape[0] > 0:

            # valid_true_boxe = [V,4]

            # shape: [13, 13, 3]
            # shape: [13, 13, 3, 4] & [V, 4] ==> [13, 13, 3, V]
            iou = calculate_iou(pred_boxes[idx], valid_true_boxes.view(-1, 4))

            best_iou = torch.max(iou, dim=-1)[0]

            ignore_mask_temp = best_iou > threshold

            ignore_mask[idx][ignore_mask_temp] = 0.

    return ignore_mask


class YoloLossLayer(nn.Module):

    def __init__(self, n_classes, image_size, device, ignore_threshold=0.5, use_focal_loss=True,  use_label_smooth=True):
        super(YoloLossLayer, self).__init__()
        self.image_size = image_size
        self.use_focal_loss = use_focal_loss
        self.use_label_smooth = use_label_smooth
        self.n_classes = n_classes
        self.device = device
        self.ignore_threshold = ignore_threshold

    def forward(self, feature_maps, y_true, anchors):

        # 13,26,52

        xy_loss = 0
        wh_loss = 0
        conf_loss = 0
        prob_loss = 0

        list_anchors = [anchors[6:9], anchors[3:6], anchors[:3]]

        for i in range(len(feature_maps)):
            xy, wh, conf, prob = self.calculate_one(
                feature_map=feature_maps[i], y_true=y_true[i], anchors=list_anchors[i])
            xy_loss += xy
            wh_loss += wh
            conf_loss += conf
            prob_loss += prob

        total_loss = xy_loss + wh_loss + conf_loss + prob_loss

        # print('conf_loss: ',conf_loss)
        # print('prob_loss: ',prob_loss)

        return total_loss, xy_loss, wh_loss, conf_loss, prob_loss

    def calculate_one(self, feature_map, y_true, anchors):


        grid_size = feature_map.shape[2]

        batch_size = y_true.shape[0]

        xy_offset, boxes, conf_logits, prob_logits, stride = predict_transform(
            feature_map, anchors, self.n_classes, self.image_size, self.device)

        object_mask = y_true[..., 4:5]

        ignore_mask = calculate_ignore_mask(
            boxes, y_true, object_mask, self.ignore_threshold)

        # ignore_mask = [batch_size, gird_size, grid_size, 3, 1]

        pred_boxes_xy = boxes[..., 0:2]
        pred_boxes_wh = boxes[..., 2:4]

        true_xy = (y_true[..., 0:2] / stride) - xy_offset
        pred_xy = (pred_boxes_xy / stride) - xy_offset

        true_tw_th = y_true[..., 2:4] / anchors

        pred_tw_th = pred_boxes_wh / anchors

        true_tw_th = torch.where(condition=(true_tw_th == 0),
                                 x=torch.ones_like(true_tw_th).type(float_tensor), other=true_tw_th)
        pred_tw_th = torch.where(condition=(pred_tw_th == 0),
                                 x=torch.ones_like(pred_tw_th).type(float_tensor), other=pred_tw_th)

        true_tw_th = torch.log(torch.clamp(true_tw_th, 1e-9, 1e9))
        pred_tw_th = torch.log(torch.clamp(pred_tw_th, 1e-9, 1e9))

        box_loss_scale = 2. - \
            (y_true[..., 2:3] / self.image_size) * \
            (y_true[..., 3:4] / self.image_size)

        xy_loss = 5. * object_mask * box_loss_scale * torch.nn.functional.mse_loss(pred_xy, true_xy , reduction = 'none')
        wh_loss = 5. * object_mask * box_loss_scale * torch.nn.functional.mse_loss(pred_tw_th, true_tw_th , reduction = 'none')

        xy_loss = torch.sum(xy_loss) / batch_size
        wh_loss = torch.sum(wh_loss) / batch_size

        conf_pos_mask = object_mask
        conf_neg_mask = (1 - object_mask) * ignore_mask


        assert (0. not in conf_logits )

        conf_loss_pos = conf_pos_mask * \
            torch.nn.functional.binary_cross_entropy_with_logits(
                target=object_mask, input=conf_logits, reduction = 'none')
        conf_loss_neg = conf_neg_mask * \
            torch.nn.functional.binary_cross_entropy_with_logits(
                target=object_mask, input=conf_logits, reduction = 'none')

        conf_loss = conf_loss_pos + 0.5 * conf_loss_neg

        if self.use_focal_loss:
            alpha = 1.0
            gamma = 2.0
            # TODO: alpha should be a mask array if needed
            focal_mask = alpha * \
                torch.pow(torch.abs(object_mask -
                                    torch.sigmoid(conf_logits)), gamma)
            conf_loss *= focal_mask
        conf_loss = torch.sum(conf_loss) / batch_size

        # shape: [N, 13, 13, 3, 1]
        # whether to use label smooth
        if self.use_label_smooth:
            delta = 0.01
            label_target = (1 - delta) * \
                y_true[..., 5:] + delta * 1. / self.n_classes
        else:
            label_target = y_true[..., 5:]
        
        # label_target = label_target.unsqueeze(-1)

        class_loss = object_mask * \
            torch.nn.functional.binary_cross_entropy_with_logits(
                input=prob_logits, target=label_target, reduction = 'none')
        class_loss = torch.sum(class_loss) / batch_size

        # print(xy_loss,wh_loss,conf_loss,class_loss)

        return xy_loss, wh_loss, conf_loss, class_loss


def predict(feature_maps, anchors, n_classes, image_size, device):

    list_anchors = [anchors[6:9], anchors[3:6], anchors[:3]]

    reorg_results = [predict_transform(feature_map, anchor, n_classes, image_size, device) for feature_map, anchor in list(
        zip(feature_maps, list_anchors))]

    def reshape(result):

        xy_offset, boxes, conf_logits, prob_logits, _ = result

        grid_size = xy_offset.shape[1]
        n_classes = prob_logits.shape[-1]

        boxes = boxes.contiguous().view(-1, grid_size * grid_size * 3, 4)
        conf_logits = conf_logits.contiguous().view(-1,
                                                    grid_size * grid_size * 3, 1)
        prob_logits = prob_logits.contiguous().view(-1,
                                                    grid_size * grid_size * 3, n_classes)

        return boxes, conf_logits, prob_logits

    boxes_list, confs_list, probs_list = [], [], []

    for result in reorg_results:

        boxes, conf, prob = reshape(result)

        conf = torch.sigmoid(conf)
        prob = torch.sigmoid(prob)

        boxes_list.append(boxes)
        confs_list.append(conf)
        probs_list.append(prob)

    boxes = torch.cat(boxes_list, dim=1)
    confs = torch.cat(confs_list, dim=1)
    probs = torch.cat(probs_list, dim=1)

    center_x, center_y, width, height = torch.split(
        boxes, [1, 1, 1, 1], dim=-1)

    x_min = center_x - width / 2
    y_min = center_y - height / 2
    x_max = center_x + width / 2
    y_max = center_y + height / 2

    boxes = torch.cat([x_min, y_min, x_max, y_max], dim=-1)

    boxes = boxes.detach().cpu().numpy()
    confs = confs.detach().cpu().numpy()
    probs = probs.detach().cpu().numpy()

    return boxes, confs, probs