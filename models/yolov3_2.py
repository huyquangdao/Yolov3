import torch.nn as nn
from utils.utils import parse_config_file
import torch
import numpy as np

if torch.cuda.is_available():
    bool_tensor = torch.cuda.BoolTensor
    float_tensor = torch.cuda.FloatTensor
else:
    bool_tensor = torch.BoolTensor
    float_tensor = torch.FloatTensor


def predict_transform(prediction, inp_dim, anchors, num_classes, device):

    batch_size = prediction.size(0)

    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride

    bbox_attrs = 5 + num_classes
    num_anchors = anchors.shape[0]

    prediction = prediction.view(
        batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(
        batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    prediction = prediction.view(
        batch_size, grid_size * grid_size, num_anchors, bbox_attrs)

    prediction = prediction.view(
        batch_size, grid_size, grid_size, num_anchors, bbox_attrs)

    box_centers, box_sizes, conf_logits, prob_logits = torch.split(
        prediction, [2, 2, 1, num_classes], dim=-1)

    rescaled_anchors = anchors / stride

    # Add the center offsets
    box_centers = torch.sigmoid(box_centers)

    grid_x = torch.arange(start=0, end=grid_size).type(float_tensor)
    grid_y = torch.arange(start=0, end=grid_size).type(float_tensor)

    grid_y, grid_x = torch.meshgrid([grid_x, grid_y])

    xy_offset = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], dim=-1)

    #xy_offset = [13,13,2]

    xy_offset = xy_offset.unsqueeze(2)

    #xy_offset = [13,13,1,2]

    box_centers = box_centers + xy_offset

    box_centers = box_centers * stride

    box_sizes = torch.exp(box_sizes) * rescaled_anchors

    box_sizes = box_sizes * stride

    boxes = torch.cat([box_centers, box_sizes], dim=-1)

    return xy_offset, boxes, conf_logits, prob_logits


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

    def forward(self, x, inp_dim, num_classes, anchors):
        return x


class Yolov3(nn.Module):
    def __init__(self, cfgfile, n_classes, image_size):
        super(Yolov3, self).__init__()
        self.blocks = parse_config_file(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
        self.image_size = image_size
        self.n_classes = n_classes

    def forward(self, x):
        modules = self.blocks[1:]
        outputs = {}  # We cache the outputs for the route layer
        write = 0  # This is explained a bit later
        output_predictions = []
        output_anchors = []
        for i, module in enumerate(modules):
            module_type = (module["type"])
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)

            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]

            elif module_type == 'yolo':

                anchors = self.module_list[i][0].anchors
                # Get the input dimensions
                inp_dim = int(self.image_size)

                # Get the number of classes
                num_classes = int(modules[i]["classes"])

                anchors = torch.from_numpy(
                    np.array(anchors)).type(float_tensor)

                output_anchors.append(anchors)

                # Transform

                output = self.module_list[i][0](
                    x, inp_dim,  num_classes, anchors)

                output_predictions.append(output)

                # if not write:  # if no collector has been intialised.
                #     detections = x
                #     write = 1
                # else:
                #     detections = torch.cat((detections, x), 1)

            outputs[i] = x

        return output_predictions, output_anchors

    def load_weights(self, weightfile):
        fp = open(weightfile, "rb")

        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            # If module_type is convolutional load weights
            # Otherwise ignore.
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if (batch_normalize):
                    bn = model[1]

                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    # Load the weights
                    bn_biases = torch.from_numpy(
                        weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(
                        weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(
                        weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(
                        weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                else:
                    # Number of biases
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(
                        weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
        conv.bias.data.copy_(conv_biases)
        num_weights = conv.weight.numel()

        # Do the same as above for weights
        conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
        ptr = ptr + num_weights

        conv_weights = conv_weights.view_as(conv.weight.data)
        conv.weight.data.copy_(conv_weights)


def calculate_iou(pred_boxes, valid_true_boxes):

    #boxes = [grid_size, grid_size, 3, 2]
    #anchors = [3,2]
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

    ignore_mask = torch.zeros(size=object_mask.shape).type(float_tensor)

    for idx in range(batch_size):

        valid_true_boxes = torch.masked_select(
            y_true[idx, ..., :4], object_mask[idx].type(bool_tensor))

        if valid_true_boxes.shape[0] > 0:

            #valid_true_boxe = [V,4]

            # shape: [13, 13, 3]
            # shape: [13, 13, 3, 4] & [V, 4] ==> [13, 13, 3, V]
            iou = calculate_iou(pred_boxes[idx], valid_true_boxes.view(-1, 4))

            best_iou = torch.max(iou, dim=-1)[0]

            ignore_mask_temp = (best_iou < threshold).type(float_tensor)

            ignore_mask[idx] = ignore_mask_temp.unsqueeze(-1)

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

        xy_offset, boxes, conf_logits, prob_logits = predict_transform(
            feature_map, anchors, self.n_classes, self.image_size, self.device)

        object_mask = y_true[..., 4:5]

        y_true_wh = y_true[..., 2:4]

        ignore_mask = calculate_ignore_mask(
            boxes, y_true, object_mask, self.ignore_threshold)

        #ignore_mask = [batch_size, gird_size, grid_size, 3, 1]

        pred_boxes_xy = boxes[..., 0:2]
        pred_boxes_wh = boxes[..., 2:4]

        true_xy = y_true[..., 0:2] / ratio - xy_offset
        pred_xy = pred_boxes_xy / ratio - xy_offset

        true_tw_th = y_true[..., 2:4] / anchors

        pred_tw_th = pred_boxes_wh / anchors

        true_tw_th = torch.where(condition=(true_tw_th == 0),
                                 x=torch.ones_like(true_tw_th).type(torch.cuda.FloatTensor), other=true_tw_th)
        pred_tw_th = torch.where(condition=(pred_tw_th == 0),
                                 x=torch.ones_like(pred_tw_th).type(torch.cuda.FloatTensor), other=pred_tw_th)

        true_tw_th = torch.log(torch.clamp(true_tw_th, 1e-9, 1e9))
        pred_tw_th = torch.log(torch.clamp(pred_tw_th, 1e-9, 1e9))

        box_loss_scale = 2. - \
            (y_true[..., 2:3] / self.image_size) * \
            (y_true[..., 3:4] / self.image_size)

        xy_loss = 5. * torch.sum(((true_xy - pred_xy)**2) *
                                 object_mask * box_loss_scale) / batch_size
        wh_loss = 5. * torch.sum(((true_tw_th - pred_tw_th)**2)
                                 * object_mask * box_loss_scale) / batch_size

        conf_pos_mask = object_mask
        conf_neg_mask = (1 - object_mask) * ignore_mask
        conf_loss_pos = conf_pos_mask * \
            torch.nn.functional.binary_cross_entropy_with_logits(
                target=object_mask, input=conf_logits)
        conf_loss_neg = 0.5 * conf_neg_mask * \
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

    reorg_results = [predict_transform(feature_map, anchor, n_classes, image_size, device) for feature_map, anchor in list(
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


def create_modules(blocks):
    # Captures the information about the input and pre-processing
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        # check the type of block
        # create a new module for the block
        # append to module_list

        # If it's a convolutional layer
        if (x["type"] == "convolutional"):
            # Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters,
                             kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            # Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # Check the activation.
            # It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)

            # If it's an upsampling layer
            # We use Bilinear2dUpsampling
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            module.add_module("upsample_{}".format(index), upsample)

        # If it is a route layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            # Start  of a route
            start = int(x["layers"][0])
            # end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            # Positive anotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index +
                                         start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        # shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        # Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1])
                       for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)
