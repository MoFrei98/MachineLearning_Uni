class Kernel:
    def __init__(self, name, is_trainable):
        self.name = name
        self.is_trainable = is_trainable

    def compute(self, x, y):
        raise NotImplementedError("Subclasses must implement this method")

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

    def get_params(self):
        return {"name": self.name, "is_trainable": self.is_trainable}