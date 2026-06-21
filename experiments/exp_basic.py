import os
import torch
from models.SC_FTMixer_SDE_Unified import Model as SC_FTMixer_SDE_Unified


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'SC_FTMixer_SDE_Unified': SC_FTMixer_SDE_Unified,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = torch.nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _acquire_device(self):
        if self.args.use_gpu:
            if self.args.use_multi_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.devices)
                device = torch.device('cuda:{}'.format(self.args.device_ids[0]))
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
                device = torch.device('cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
        return device

    def _get_data(self, flag):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
