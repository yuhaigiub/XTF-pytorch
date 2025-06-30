import torch
from ..runners import SimpleTimeSeriesForecastingRunner


class MSTERunner(SimpleTimeSeriesForecastingRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)

    def forward(self, data, epoch=None, iter_num=None, train=True, **kwargs):
        model_return = super().forward(data, epoch, iter_num, train, **kwargs)

        model_return['model'] = self.model if train == True else None
        return model_return
