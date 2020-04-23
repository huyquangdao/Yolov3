from base.inference import BaseInference
from utils.utils import cpu_nms
from models.yolov3 import predict


class YoloInference(BaseInference):

    def __init__(self, model, device, n_classes, anchors):
        super().__init__(model, device)
        self.anchors = anchors
        self.n_classes = n_classes

        self.anchors = self.anchors.to(self.device)

    def inference(self, input, iou_threshold=0.45, score_threshold=0.3, top_k=200):
        if len(input.shape) < 4:
            input = input.unsqueeze(0)

        image_size = input.shape[1]

        input = input.permute(0, 3, 1, 2).to(self.device)
        output = self.model(input)
        boxes, confidence_scores, class_scores = predict(output, self.anchors, self.n_classes,image_size,self.device)
        scores = confidence_scores * class_scores
        boxes_, scores_, labels_ = cpu_nms(boxes=boxes, scores=scores, num_classes=self.n_classes,
                        max_boxes=top_k, score_thresh=score_threshold, iou_thresh=iou_threshold)
        return boxes_, scores_, labels_
