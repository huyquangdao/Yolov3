
class BaseMeters:

    def __init__(self):
        self.memory = []

    def write(self, number):
        self.memory.append(number)

    def clear(self):
        self.memory = []

    def average(self):
        return sum(self.memory) / len(self.memory)
