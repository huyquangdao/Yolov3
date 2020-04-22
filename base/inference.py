
class BaseInference:

    def __init__(self, model, device):
        self.model = model
        self.device = device

        self.model.to(self.device)

    def inference(self, input):
        raise NotImplementedError('You must implement this method')
