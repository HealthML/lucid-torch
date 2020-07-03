from lucid_torch.objectives import Objective


class ImageObjective(Objective):
    def __init__(self):
        super().__init__(None)

    def register(self, model):
        pass

    def remove_hook(self):
        pass
