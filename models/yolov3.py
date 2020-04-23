import torch.nn as nn
import torch

if torch.cuda.is_available():
    bool_tensor = torch.cuda.BoolTensor()
    float_tensor = torch.cuda.FloatTensor()
else:
    bool_tensor = torch.BoolTensor()
    float_tensor = torch.FloatTensor()


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

        output = self.conv_bn_relu1(input)

        output = self.conv_bn_relu2(output)

        output = output + input

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
            in_channels=1024, out_channels=3 * (5 + self.n_classes), kernel_size=1)
        self.upsample1 = UpsampleLayer(
            scale_factor=2, in_channels=512, out_channels=256, k_size=1, stride=1, padding=True)

        self.yolo_layer2 = YoloLayer(in_channels=768, out_channels=512)
        self.conv2 = nn.Conv2d(
            in_channels=512, out_channels=3 * (5 + self.n_classes), kernel_size=1)
        self.upsample2 = UpsampleLayer(
            scale_factor=2, in_channels=256, out_channels=128, k_size=1, stride=1, padding=True)

        self.yolo_layer3 = YoloLayer(in_channels=384, out_channels=256)
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=3 * (5 + self.n_classes), kernel_size=1)

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


def reorg_layer(feature_map, anchors, n_classes, image_size, device):

    grid_size = feature_map.shape[2]
    ratio = image_size / grid_size

    rescaled_anchors = anchors / ratio

    feature_map = feature_map.permute(0, 2, 3, 1)

    feature_map = feature_map.view(-1, grid_size, grid_size, 3, 5 + n_classes)

    box_centers, box_sizes, conf_logits, prob_logits = torch.split(
        feature_map, [2, 2, 1, n_classes], dim=-1)

    box_centers = torch.sigmoid(box_centers)

    grid_x = torch.arange(start=0, end=grid_size).to(device)
    grid_y = torch.arange(start=0, end=grid_size).to(device)

    grid_y, grid_x = torch.meshgrid([grid_x, grid_y])

    grid_y = grid_y.contiguous().view(-1, 1)

    grid_x = grid_x.contiguous().view(-1, 1)

    xy_offset = torch.cat([grid_x, grid_y], dim=-1)

    xy_offset = xy_offset.view(grid_size, grid_size, 1, 2)

    box_centers = box_centers + xy_offset

    box_centers = box_centers * ratio

    box_sizes = torch.exp(box_sizes) * rescaled_anchors

    box_sizes = box_sizes * ratio

    boxes = torch.cat([box_centers, box_sizes], dim=-1)

    return xy_offset, boxes, conf_logits, prob_logits


def calculate_iou(boxes_wh, anchors):

    #boxes = [grid_size, grid_size, 3, 2]
    #anchors = [3,2]
    min_hw = torch.max(- boxes_wh / 2, - anchors / 2)
    max_hw = torch.min(boxes_wh/2, anchors / 2)

    whs = max_hw - min_hw

    iou = (whs[..., 0] * whs[..., 1]) / (
        boxes_wh[..., 0] * boxes_wh[..., 1] + anchors[:, 0] * anchors[:, 1] - whs[..., 0] * whs[...,
                                                                                                1] + 1e-10)
    #iou = [grid_size, grid_size ,3]

    return iou


def calculate_ignore_mask(object_mask, y_true_boxes, anchors, threshold):

    ignore_mask = torch.ones(size=object_mask.shape).type(
        torch.cuda.FloatTensor)
    #ignore_mask = [batch_size, grid_size, grid_size, 3,1]

    temp = y_true_boxes.clone()
    #temp = [batch_size, grid_size, grid_size, 3, 2]

    batch_size = object_mask.shape[0]
    for i in range(batch_size):
        list_boxes_coord = (y_true_boxes[i] != 0).nonzero()
        for coor in list_boxes_coord:
            dy, dx, db, _, = coor
            temp[i, dy, dx, :] = y_true_boxes[i, dy, dx, db]

        iou = calculate_iou(temp[i], anchors)

        ignore_mask[i] = ignore_mask[i].masked_fill(iou.unsqueeze(-1) > 0.5, 0)

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
        return total_loss, xy_loss, wh_loss, conf_loss, prob_loss

    def calculate_one(self, feature_map, y_true, anchors):

        grid_size = feature_map.shape[2]

        batch_size = y_true.shape[0]

        ratio = self.image_size / grid_size

        xy_offset, boxes, conf_logits, prob_logits = reorg_layer(
            feature_map, anchors, self.n_classes, self.image_size, self.device)

        object_mask = y_true[..., 4:5]

        y_true_wh = y_true[..., 2:4]

        ignore_mask = calculate_ignore_mask(
            object_mask, y_true_wh, anchors, self.ignore_threshold)

        #ignore_mask = [batch_size, gird_size, grid_size, 3, 1]

        pred_boxes_xy = boxes[..., 0:2]
        pred_boxes_wh = boxes[..., 2:4]

        true_xy = y_true[..., 0:2] / ratio - xy_offset
        pred_xy = pred_boxes_xy / ratio - xy_offset

        true_tw_th = y_true[..., 2:4] / anchors

        pred_tw_th = pred_boxes_wh / anchors

        # true_tw_th = torch.where(condition=(true_tw_th == 0),
        #                       x= torch.ones_like(true_tw_th).type(torch.cuda.FloatTensor), other= true_tw_th)
        # true_tw_th = torch.where(condition=(pred_tw_th == 0),
        #                       x= torch.ones_like(pred_tw_th).type(torch.cuda.FloatTensor), other= pred_tw_th)

        true_tw_th = torch.log(torch.clamp(true_tw_th, 1e-9, 1e9))
        pred_tw_th = torch.log(torch.clamp(pred_tw_th, 1e-9, 1e9))

        box_loss_scale = 2. - \
            (y_true[..., 2:3] / self.image_size) * \
            (y_true[..., 3:4] / self.image_size)

        xy_loss = torch.sum(((true_xy - pred_xy)**2) *
                            object_mask * box_loss_scale) / batch_size
        wh_loss = torch.sum(((true_tw_th - pred_tw_th)**2)
                            * object_mask * box_loss_scale) / batch_size

        conf_pos_mask = object_mask
        conf_neg_mask = (1 - object_mask) * ignore_mask
        conf_loss_pos = conf_pos_mask * \
            torch.nn.functional.binary_cross_entropy_with_logits(
                target=object_mask, input=conf_logits)
        conf_loss_neg = conf_neg_mask * \
            torch.nn.functional.binary_cross_entropy_with_logits(
                target=object_mask, input=conf_logits)

        conf_loss = conf_loss_pos + conf_loss_neg

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
        class_loss = object_mask * \
            torch.nn.functional.binary_cross_entropy_with_logits(
                input=prob_logits, target=label_target)
        class_loss = torch.sum(class_loss) / batch_size

        return xy_loss, wh_loss, conf_loss, class_loss


def predict(feature_maps, anchors, n_classes, image_size, device):

    list_anchors = [anchors[6:], anchors[3:6], anchors[:3]]

    reorg_results = [reorg_layer(feature_map, anchor, n_classes, image_size, device) for feature_map, anchor in list(
        zip(feature_maps, list_anchors))]

    def reshape(result):

        xy_offset, boxes, conf_logits, prob_logits = result

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

    return boxes, confs, probs
