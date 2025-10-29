import enum
import math

import torch
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .m2no import M2NO2d
from .physics_loss import rho


class LambdaSched(nn.Module):
    """
    Return lambda(t) with shape [N] or [1] (broadcastable).
    t: [N] int64 timesteps (0..T-1), reverse sampling goes T-1 -> 0
    """
    def __init__(self, T: int,
                 mode: str = "cosine",
                 lam_min: float = 0.1,
                 lam_max: float = 1.0,
                 plateau_ratio: float = 0.4,   # for piecewise
                 warmup_ratio: float = 0.2,    # for piecewise
                 poly_gamma: float = 2.0,      # for poly
                 learned: bool = False,
                 emb_dim: int = 64):
        super().__init__()
        self.T = T
        self.mode = mode
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.plateau_ratio = plateau_ratio
        self.warmup_ratio = warmup_ratio
        self.poly_gamma = poly_gamma

        # learned head (optional)
        self.learned = learned
        if learned:
            self.time_mlp = nn.Sequential(
                nn.Linear(emb_dim, 4*emb_dim), nn.SiLU(),
                nn.Linear(4*emb_dim, 1)
            )
            # sin-cos embedding
            self.emb_dim = emb_dim

    def timestep_embed(self, t: torch.Tensor):
        """ sinusoidal TE, t in [0,T-1] -> [N, emb_dim] """
        half = self.emb_dim // 2
        freqs = torch.exp(
            torch.arange(0, half, device=t.device, dtype=torch.float32)
            * (-math.log(10000.0) / half)
        )  # [half]
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [N,half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # [N,emb_dim]
        if emb.shape[1] < self.emb_dim:  # odd
            emb = torch.nn.functional.pad(emb, (0, self.emb_dim - emb.shape[1]))
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [N] int64, 0..T-1 (we assume reverse goes from T-1 -> 0)
        Return: [N] float32 in [lam_min, lam_max]
        """
        device = t.device
        T = max(int(self.T), 1)
        # normalize to [0,1], where 0=early (coarse), 1=late (fine)
        # we want lambda big early, smaller later -> use s = 1 - t/T
        s = 1.0 - (t.float() / max(T-1, 1))

        if self.mode == "cosine":
            # smooth ramp-up then mild decay
            # base in [0,1]: 0.5*(1 - cos(pi*s))  (0->0, 1->1)
            base = 0.5 * (1.0 - torch.cos(math.pi * s))
            # apply a gentle decay near the end
            decay = 0.5 * (1.0 + torch.cos(math.pi * (t.float() / max(T-1,1))))
            lam = base * (0.7 + 0.3 * decay)

        elif self.mode == "linear":
            # simple monotone from high (early) to low (late)
            lam = 0.5 + 0.5 * s  # [0,1] -> [0.5,1.0]

        elif self.mode == "piecewise":
            w = self.warmup_ratio
            p = self.plateau_ratio
            # segments on s in [0,1]
            lam = torch.empty_like(s)
            s1 = (s < w)
            s2 = (~s1) & (s < w + p)
            s3 = ~(s1 | s2)
            lam[s1] = s[s1] / max(w, 1e-6)                  # 0 -> 1
            lam[s2] = 1.0                                   # plateau
            lam[s3] = torch.clamp(1.0 - (s[s3] - (w+p)) / max(1.0 - (w+p), 1e-6), 0.0, 1.0) * 0.7 + 0.3
            # normalize to [0,1]
            lam = lam

        elif self.mode == "poly":
            # polynomial ramp-up then keep high
            lam = s.pow(self.poly_gamma)

        else:
            # default fallback
            lam = s  # [0,1]

        # learned refinement (optional), bounded to [lam_min, lam_max]
        if self.learned:
            emb = self.timestep_embed(t)           # [N,emb_dim]
            delta = self.time_mlp(emb).squeeze(1)  # [N]
            lam = lam + 0.1 * torch.tanh(delta)

        lam = torch.clamp(lam, 0.0, 1.0)
        lam = self.lam_min + (self.lam_max - self.lam_min) * lam
        return lam


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, beta_start, beta_end):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        return np.linspace(
            beta_start**0.5, beta_end**0.5, num_diffusion_timesteps, dtype=np.float64
        )**2
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def get_named_eta_schedule(
        schedule_name,
        num_diffusion_timesteps,
        min_noise_level,
        etas_end=0.99,
        kappa=1.0,
        kwargs=None):
    """
    Get a pre-defined eta schedule for the given name.

    The eta schedule library consists of eta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    """
    if schedule_name == 'exponential':
        # ponential = kwargs.get('ponential', None)
        # start = math.exp(math.log(min_noise_level / kappa) / ponential)
        # end = math.exp(math.log(etas_end) / (2*ponential))
        # xx = np.linspace(start, end, num_diffusion_timesteps, endpoint=True, dtype=np.float64)
        # sqrt_etas = xx**ponential
        power = kwargs.get('power', None)
        # etas_start = min(min_noise_level / kappa, min_noise_level, math.sqrt(0.001))
        etas_start = min(min_noise_level / kappa, min_noise_level)
        increaser = math.exp(1/(num_diffusion_timesteps-1)*math.log(etas_end/etas_start))
        base = np.ones([num_diffusion_timesteps, ]) * increaser
        power_timestep = np.linspace(0, 1, num_diffusion_timesteps, endpoint=True)**power
        power_timestep *= (num_diffusion_timesteps-1)
        sqrt_etas = np.power(base, power_timestep) * etas_start
    elif schedule_name == 'ldm':
        import scipy.io as sio
        mat_path = kwargs.get('mat_path', None)
        sqrt_etas = sio.loadmat(mat_path)['sqrt_etas'].reshape(-1)
    else:
        raise ValueError(f"Unknow schedule_name {schedule_name}")

    return sqrt_etas


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon
    PREVIOUS_X = enum.auto()  # the model predicts epsilon
    RESIDUAL = enum.auto()  # the model predicts epsilon
    EPSILON_SCALE = enum.auto()  # the model predicts epsilon


class LossType(enum.Enum):
    MSE = enum.auto()           # simplied MSE
    WEIGHTED_MSE = enum.auto()  # weighted mse derived from KL


class ModelVarTypeDDPM(enum.Enum):
    """
    What is used as the model's output variance.
    """

    LEARNED = enum.auto()
    LEARNED_RANGE = enum.auto()
    FIXED_LARGE = enum.auto()
    FIXED_SMALL = enum.auto()


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    :param sqrt_etas: a 1-D numpy array of etas for each diffusion timestep,
                starting at T and going to 1.
    :param kappa: a scaler controling the variance of the diffusion kernel
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param loss_type: a LossType determining the loss function to use.
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    :param scale_factor: a scaler to scale the latent code
    :param sf: super resolution factor
    """

    def __init__(
        self,
        *,
        sqrt_etas,
        kappa,
        model_mean_type,
        loss_type,
        m2no_params=None,
        sf=4,
        scale_factor=None,
        normalize_input=True,
        latent_flag=True,
    ):
        self.kappa = kappa
        self.model_mean_type = model_mean_type
        self.loss_type = loss_type
        self.scale_factor = scale_factor
        self.normalize_input = normalize_input
        self.latent_flag = latent_flag
        self.sf = sf

        # Use float64 for accuracy.
        self.sqrt_etas = sqrt_etas
        self.etas = sqrt_etas**2
        assert len(self.etas.shape) == 1, "etas must be 1-D"
        assert (self.etas > 0).all() and (self.etas <= 1).all()

        self.num_timesteps = int(self.etas.shape[0])
        self.etas_prev = np.append(0.0, self.etas[:-1])
        self.alpha = self.etas - self.etas_prev

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = kappa**2 * self.etas_prev / self.etas * self.alpha
        self.posterior_variance_clipped = np.append(
                self.posterior_variance[1], self.posterior_variance[1:]
                )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(self.posterior_variance_clipped)
        self.posterior_mean_coef1 = self.etas_prev / self.etas
        self.posterior_mean_coef2 = self.alpha / self.etas

        # weight for the mse loss
        if model_mean_type in [ModelMeanType.START_X, ModelMeanType.RESIDUAL]:
            weight_loss_mse = 0.5 / self.posterior_variance_clipped * (self.alpha / self.etas)**2
        elif model_mean_type in [ModelMeanType.EPSILON, ModelMeanType.EPSILON_SCALE]  :
            weight_loss_mse = 0.5 / self.posterior_variance_clipped * (
                    kappa * self.alpha / ((1-self.etas) * self.sqrt_etas)
                    )**2
        else:
            raise NotImplementedError(model_mean_type)

        # self.weight_loss_mse = np.append(weight_loss_mse[1],  weight_loss_mse[1:])
        self.weight_loss_mse = weight_loss_mse
        
        self.S = M2NO2d(m2no_params)
        
        self.lambda_sched = LambdaSched(
            T=self.num_timesteps,
            mode='cosine',
            lam_min=0.2,
            lam_max=1.0,
        )
        

    def q_sample(self, x_start, y, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.etas, t, x_start.shape) * (y - x_start) + x_start
            + _extract_into_tensor(self.sqrt_etas * self.kappa, t, x_start.shape) * noise
        )
    
    def residual(self, u, u0, t, R_op=None):
        """
        u  : current HR estimate (x_t)
        u0 : coarse / LR-aligned field (u_0)
        t  : timestep tensor [N]
        R_op: optional restriction; if None, identity

        returns: r = (u0 - R(u)) + lambda(t) * rho(u)
        """
        # 1) restriction consistency
        if R_op is None:
            Ru = u
        else:
            Ru = R_op(u)          # shape must match u0

        r_data = u0 - Ru          # NOTE: sign and restriction aligned to paper

        # 2) physics residual on current estimate
        r_phys = rho(
            u=u,
            t=t,
            u0_for_anchor=u0,               
            w_biharm=0.5, w_aniso=0.5, w_spec=0.5,       # 默认组合（稳）
            w_noflux=0.0,   # 有岸线时打开
            range_lo=None, range_hi=None                  # 若有物理范围就填
        )

        # 3) time schedule (scalar or per-sample)
        lam = self.lambda_sched(t)  # shape: [N] or broadcastable to r_phys

        # 4) (可选) 尺度平衡，避免 r_phys 量纲偏大/偏小
        # 估计每个 batch 的标准差，做一次性标准化以稳定训练
        eps = 1e-6
        s_data = (r_data.pow(2).mean(dim=(1,2,3)) + eps).sqrt().view(-1,1,1,1)
        s_phys = (r_phys.pow(2).mean(dim=(1,2,3)) + eps).sqrt().view(-1,1,1,1)
        r_phys_balanced = r_phys * (s_data / (s_phys + eps))

        # 5) 组合
        r = r_data + lam.view(-1,1,1,1) * r_phys_balanced
        return r
    
    def q_sample_consistent(self, u_0, u, t, noise=None, kappa=None):
        if noise is None:
            noise = th.randn_like(u_0)
        
        # r_data = u - u_0
        # r_phys = self.rho(u_0)
        # lam = self.lambda_sched(t)
        # r = r_data + lam * r_phys
        r = self.residual(u=u, u0=u_0, t=t)
        self.S.to(u_0.device)
        e = self.S(u_0, r, t)
        
        alpha_t = _extract_into_tensor(self.etas, t, u_0.shape)
        sqrt_alpha_t = _extract_into_tensor(self.sqrt_etas, t, u_0.shape)
        kappa = self.kappa if kappa is None else kappa
        sigma_t = sqrt_alpha_t * kappa
        
        u_t = u_0 + alpha_t * e + sigma_t * noise
        
        return u_t

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_t
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_start
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance_consistent(
        self, model, x_t, y, t,
        clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x_t.shape[:2]
        assert t.shape == (B,)

        # 1) 常规网络前向
        model_output = model(self._scale_input(x_t, t), t, **model_kwargs)

        # 2) 系数表
        model_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)

        # 3) 定义一个内部包装，包含 D/S/物理启发
        def enhance_residual_with_DS(x_cur, y_obs, raw_residual):
            # 数据一致残差（HR->LR）
            r_data = y_obs - self.D(x_cur)             # 关键：把当前 HR 估计投到 LR 域再比对
            # 物理启发（如无散度/频谱等的梯度方向；没实现时置 0）
            r_phys = getattr(self, "rho", lambda z: 0.0)(x_cur)
            r = r_data + self.lambda_phys * r_phys
            # 预条件/细化（MG + refine）
            e = self.S(x_cur, r, t)                    # 与 raw_residual 同域/同形状
            # 融合：网络预测 residual + 结构化校正（权重你也可做成 schedule）
            return raw_residual + e

        # 4) 三种参数化下的 x0 预测
        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:
            # 1) 模型直接给出 x0 估计
            pred_xstart_0 = model_output

            # 2) 构造一致性残差 + 物理启发
            # r_data = y - self.D(pred_xstart_0)                    # HR->LR 数据一致
            # r_data = y - pred_xstart_0
            # # r_phys = getattr(self, "rho", lambda z: 0.0)(pred_xstart_0)  # 可为0
            # r = r_data
            r = self.residual(u=pred_xstart_0, u0=y, t=t)

            # 3) 预条件/细化方向（MG + Refine）
            e = self.S(pred_xstart_0, r, t)

            # 4) 小步校正（避免破坏扩散后验的几何）
            alpha_corr = _extract_into_tensor(self.etas, t, pred_xstart_0.shape) * 0.25
            pred_xstart = pred_xstart_0 + alpha_corr * e

            # 5) 可选：物理投影 + 裁剪（沿用你原来的 denoised_fn / clip_denoised）
            if denoised_fn is not None:
                pred_xstart = denoised_fn(pred_xstart)
            if clip_denoised:
                pred_xstart = pred_xstart.clamp(-1, 1)
        elif self.model_mean_type == ModelMeanType.RESIDUAL:
            # 原生 residual 参数化：在还原 x0 前，加上 D/S 增强
            # raw_resid 由网络给出，形状与 x_t 相容
            raw_resid = model_output
            resid = enhance_residual_with_DS(x_t, y, raw_resid)
            pred_xstart = process_xstart(
                self._predict_xstart_from_residual(y=y, residual=resid)
            )

        elif self.model_mean_type in (ModelMeanType.EPSILON, ModelMeanType.EPSILON_SCALE):
            # 噪声参数化：先还原出 x0，再用 D/S 做一次“后校正”（更保守的接入方式）
            pred_xstart = self._predict_xstart_from_eps(x_t=x_t, y=y, t=t, eps=model_output) \
                        if self.model_mean_type == ModelMeanType.EPSILON else \
                        self._predict_xstart_from_eps_scale(x_t=x_t, y=y, t=t, eps=model_output)
            # 在像素/物理域做一次小步修正（可选，步长用很小的 alpha_hat）
            alpha_hat = _extract_into_tensor(self.etas, t, x_t.shape) * 0.25
            r_data = y - self.D(pred_xstart)
            r_phys = getattr(self, "rho", lambda z: 0.0)(pred_xstart)
            e = self.S(pred_xstart, r_data + self.lambda_phys * r_phys, t)
            pred_xstart = process_xstart(pred_xstart + alpha_hat * e)

        else:
            raise ValueError(f'Unknown Mean type: {self.model_mean_type}')

        # 5) 标准扩散后验（保持不变）
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x_t, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x_t.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    
    def p_mean_variance(
        self, model, x_t, y, t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x_t: the [N x C x ...] tensor at time t.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        model_output = model(self._scale_input(x_t, t), t, **model_kwargs)

        model_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:      # predict x_0
            pred_xstart = process_xstart(model_output)
        elif self.model_mean_type == ModelMeanType.RESIDUAL:      # predict x_0
            pred_xstart = process_xstart(
                self._predict_xstart_from_residual(y=y, residual=model_output)
                )
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x_t, y=y, t=t, eps=model_output)
            )                                                  #  predict \eps
        elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps_scale(x_t=x_t, y=y, t=t, eps=model_output)
            )                                                  #  predict \eps
        else:
            raise ValueError(f'Unknown Mean type: {self.model_mean_type}')

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x_t, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x_t.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, y, t, eps):
        assert x_t.shape == eps.shape
        return  (
            x_t - _extract_into_tensor(self.sqrt_etas, t, x_t.shape) * self.kappa * eps
                - _extract_into_tensor(self.etas, t, x_t.shape) * y
        ) / _extract_into_tensor(1 - self.etas, t, x_t.shape)

    def _predict_xstart_from_eps_scale(self, x_t, y, t, eps):
        assert x_t.shape == eps.shape
        return  (
            x_t - eps - _extract_into_tensor(self.etas, t, x_t.shape) * y
        ) / _extract_into_tensor(1 - self.etas, t, x_t.shape)

    def _predict_xstart_from_residual(self, y, residual):
        assert y.shape == residual.shape
        return (y - residual)

    def _predict_eps_from_xstart(self, x_t, y, t, pred_xstart):
        return (
            x_t - _extract_into_tensor(1 - self.etas, t, x_t.shape) * pred_xstart
                - _extract_into_tensor(self.etas, t, x_t.shape) * y
        ) / _extract_into_tensor(self.kappa * self.sqrt_etas, t, x_t.shape)

    def p_sample(self, model, x, y, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, noise_repeat=False):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_t.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        # out = self.p_mean_variance(
        #     model,
        #     x,
        #     y,
        #     t,
        #     clip_denoised=clip_denoised,
        #     denoised_fn=denoised_fn,
        #     model_kwargs=model_kwargs,
        # )
        
        out = self.p_mean_variance_consistent(
            model,
            x,
            y,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        
        noise = th.randn_like(x)
        if noise_repeat:
            noise = noise[0,].repeat(x.shape[0], 1, 1, 1)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "mean":out["mean"]}

    def p_sample_loop(
        self,
        y,
        model,
        first_stage_model=None,
        consistencydecoder=None,
        noise=None,
        noise_repeat=False,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param y: the [N x C x ...] tensor of degraded inputs.
        :param model: the model module.
        :param first_stage_model: the autoencoder model
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            y,
            model,
            first_stage_model=first_stage_model,
            noise=noise,
            noise_repeat=noise_repeat,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample["sample"]
        with th.no_grad():
            out = self.decode_first_stage(
                    final,
                    first_stage_model=first_stage_model,
                    consistencydecoder=consistencydecoder,
                    )
        return out

    def p_sample_loop_progressive(
            self, y, model,
            first_stage_model=None,
            noise=None,
            noise_repeat=False,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        z_y = self.encode_first_stage(y, first_stage_model, up_sample=False)

        # generating noise
        if noise is None:
            noise = th.randn_like(z_y)
        if noise_repeat:
            noise = noise[0,].repeat(z_y.shape[0], 1, 1, 1)
        z_sample = self.prior_sample(z_y, noise)

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * y.shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    z_sample,
                    z_y,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    noise_repeat=noise_repeat,
                )
                yield out
                z_sample = out["sample"]

    def decode_first_stage(self, z_sample, first_stage_model=None, consistencydecoder=None):
        batch_size = z_sample.shape[0]
        data_dtype = z_sample.dtype

        if consistencydecoder is None:
            pass
            # model = first_stage_model
            # decoder = first_stage_model.decode
            # model_dtype = next(model.parameters()).dtype
        else:
            model = consistencydecoder
            decoder = consistencydecoder
            model_dtype = next(model.ckpt.parameters()).dtype

        if first_stage_model is None:
            return z_sample
        else:
            z_sample = 1 / self.scale_factor * z_sample
            if consistencydecoder is None:
                out = decoder(z_sample.type(model_dtype))
            else:
                with th.cuda.amp.autocast():
                    out = decoder(z_sample)
            if not model_dtype == data_dtype:
                out = out.type(data_dtype)
            return out

    def encode_first_stage(self, y, first_stage_model, up_sample=False):
        data_dtype = y.dtype
        # model_dtype = next(first_stage_model.parameters()).dtype
        if up_sample and self.sf != 1:
            y = F.interpolate(y, scale_factor=self.sf, mode='bicubic')
        if first_stage_model is None:
            return y
        else:
            if not model_dtype == data_dtype:
                y = y.type(model_dtype)
            with th.no_grad():
                z_y = first_stage_model.encode(y)
                out = z_y * self.scale_factor
            if not model_dtype == data_dtype:
                out = out.type(data_dtype)
            return out

    def prior_sample(self, y, noise=None):
        """
        Generate samples from the prior distribution, i.e., q(x_T|x_0) ~= N(x_T|y, ~)

        :param y: the [N x C x ...] tensor of degraded inputs.
        :param noise: the [N x C x ...] tensor of degraded inputs.
        """
        if noise is None:
            noise = th.randn_like(y)

        t = th.tensor([self.num_timesteps-1,] * y.shape[0], device=y.device).long()

        return y + _extract_into_tensor(self.kappa * self.sqrt_etas, t, y.shape) * noise

    def training_losses(
            self, model, x_start, y, t,
            first_stage_model=None,
            model_kwargs=None,
            noise=None,
            ):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param first_stage_model: autoencoder model
        :param x_start: the [N x C x ...] tensor of inputs.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :param up_sample_lq: Upsampling low-quality image before encoding
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}

        z_y = self.encode_first_stage(y, first_stage_model, up_sample=False)
        z_start = self.encode_first_stage(x_start, first_stage_model, up_sample=False)

        if noise is None:
            noise = th.randn_like(z_start)

        # z_t = self.q_sample(z_start, z_y, t, noise=noise)
        z_t = self.q_sample_consistent(z_start, z_y, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.MSE or self.loss_type == LossType.WEIGHTED_MSE:
            from utils.loss import LpLoss
            loss_fun = LpLoss()
            model_output = model(self._scale_input(z_t, t), t, **model_kwargs)
            target = {
                ModelMeanType.START_X: z_start,
                ModelMeanType.RESIDUAL: z_y - z_start,
                ModelMeanType.EPSILON: noise,
                ModelMeanType.EPSILON_SCALE: noise*self.kappa*_extract_into_tensor(self.sqrt_etas, t, noise.shape),
            }[self.model_mean_type]
            assert model_output.shape == target.shape == z_start.shape
            terms["mse"] = loss_fun(model_output, target)
            if self.model_mean_type == ModelMeanType.EPSILON_SCALE:
                terms["mse"] /= (self.kappa**2 * _extract_into_tensor(self.etas, t, t.shape))
            if self.loss_type == LossType.WEIGHTED_MSE:
                weights = _extract_into_tensor(self.weight_loss_mse, t, t.shape)
            else:
                weights = 1
            terms["mse"] *= weights
        else:
            raise NotImplementedError(self.loss_type)

        if self.model_mean_type == ModelMeanType.START_X:      # predict x_0
            pred_zstart = model_output
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_zstart = self._predict_xstart_from_eps(x_t=z_t, y=z_y, t=t, eps=model_output)
        elif self.model_mean_type == ModelMeanType.RESIDUAL:
            pred_zstart = self._predict_xstart_from_residual(y=z_y, residual=model_output)
        elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
            pred_zstart = self._predict_xstart_from_eps_scale(x_t=z_t, y=z_y, t=t, eps=model_output)
        else:
            raise NotImplementedError(self.model_mean_type)

        return terms, z_t, pred_zstart

    def _scale_input(self, inputs, t):
        if self.normalize_input:
            if self.latent_flag:
                # the variance of latent code is around 1.0
                std = th.sqrt(_extract_into_tensor(self.etas, t, inputs.shape) * self.kappa**2 + 1)
                inputs_norm = inputs / std
            else:
                inputs_max = _extract_into_tensor(self.sqrt_etas, t, inputs.shape) * self.kappa * 3 + 1
                inputs_norm = inputs / inputs_max
        else:
            inputs_norm = inputs
        return inputs_norm
