from .base import BaseForecaster
from models import _ddpm_dict


class DDPMForecaster(BaseForecaster):
    def __init__(self, path):
        super().__init__(path)
        
    def build_model(self, **kwargs):
        self.beta_schedule = self.args['beta_schedule']
        model = _ddpm_dict[self.model_name]["model"](self.model_args)
        diffusion = _ddpm_dict[self.model_name]["diffusion"](
            model,
            model_args=self.model_args,
            schedule_opt=self.beta_schedule['train']
        )
        diffusion.set_new_noise_schedule(
            self.beta_schedule['train'],
            device=self.device
        )
        diffusion.set_loss(self.device)

        return diffusion

    def inference(self, x, y, **kwargs):
        x = x.permute(0, 3, 1, 2)
        y_pred = self.model.super_resolution(x, continous=False)
        y_pred = y_pred.permute(0, 2, 3, 1).reshape(y.shape)
        return y_pred
