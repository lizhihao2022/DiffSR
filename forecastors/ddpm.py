from .base import BaseForecaster
from models.ddpm import UNet, GaussianDiffusion


class DDPMForecaster(BaseForecaster):
    def __init__(self, path):
        self.beta_schedule = args['beta_schedule']
        super().__init__(path)
        
    def build_model(self, **kwargs):
        model = UNet(
            in_channel=self.model_args['in_channel'],
            out_channel=self.model_args['out_channel'],
            inner_channel=self.model_args['inner_channel'],
            channel_mults=self.model_args['channel_mults'],
            attn_res=self.model_args['attn_res'],
            res_blocks=self.model_args['res_blocks'],
            dropout=self.model_args['dropout'],
            image_size=self.model_args['image_size'],
            )
        diffusion = GaussianDiffusion(
            model,
            image_size=self.model_args['image_size'],
            channels=self.model_args['channels'],
            loss_type=self.model_args['loss_type'],
            conditional=self.model_args['conditional'],
            schedule_opt=self.beta_schedule['train'],
            )
        
        diffusion.set_new_noise_schedule(
            self.beta_schedule['train'],
            device=self.device
        )
        diffusion.set_loss(self.device)

        return diffusion
