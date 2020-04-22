import torch
from base.inference import BaseInference

class2label = {
    0:'dog',
    1:'cat'
}

class DogCatInference(BaseInference):

    def __init__(self, model, device, transform = None):
        super(DogCatInference,self).__init__(model,device)
        self.transform = transform

    def inference(self, input_tensor):
        if self.transform is not None:
            input_tensor = self.transform(input_tensor)
        input_tensor = input_tensor.to(self.device)

        if len(input_tensor.shape) < 4:
            input_tensor = input_tensor.unsqueeze(0)

        logits = self.model(input_tensor)
        classes = torch.argmax(logits,dim=-1).detach().cpu().numpy().tolist()
        result = []
        for clas in classes:
            result.append(class2label[int(clas)])
        return result
