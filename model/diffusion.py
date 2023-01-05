# DiffusionModel 
# UNet.py -> AnoDDPM
import torch.nn as nn
import torch.nn.functional as F
import torch
from model.core.utils import *
from model.core.solver import *
import model.core.evaluation as evaluation
from model.core.simplex import Simplex_noise
import os
import matplotlib.pyplot as plt

class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # parameters
        LOSS_TYPE="l2"
        noise= "simplex" #guass
        img_channels=3
        T= 1000
        BETA_SCHEDULE = "linear"
        in_channels = 3
        betas = get_beta_schedule(T,BETA_SCHEDULE)
        loss_weight='none'
        img_size=256
        
        if noise == "gauss":
            self.noise_fn = lambda x,t : torch.randn_like(x)
        else:
            self.simplex = Simplex_noise()
            self.noise_fn = lambda x,t : generate_simplex_noise(self.simplex,x,t,False,in_channels=img_channels)
        
        self.img_size = img_size
        self.img_channels = in_channels
        self.loss_type = LOSS_TYPE
        self.num_timesteps = len(betas)
        self.weights = np.ones(self.num_timesteps)

        if loss_weight == 'prop-t':
            self.weights = np.arange(self.num_timesteps, 0, -1)
        elif loss_weight == "uniform":
            self.weights = np.ones(self.num_timesteps)

        self.loss_weight = loss_weight

        alphas = 1. - betas
        self.betas = betas
        self.sqrt_alphas = np.sqrt(alphas)
        self.sqrt_betas = np.sqrt(betas)

        self.alphas_cumprod = np.cumprod(alphas,axis=0)
        self.alphas_cumprod_prev = np.append(1.0,self.alphas_cumprod[:-1])

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev)/(1.0-self.alphas_cumprod))
        self.posterior_variance_clipped = np.log(np.append(self.posterior_variance[1],self.posterior_variance[1:]))

        self.posterior_mean_coef1 = (betas*np.sqrt(self.alphas_cumprod_prev)/(1.0 - self.alphas_cumprod))

        self.posterior_mean_coef2 = ((1.0)-self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = np.log(
                np.append(self.posterior_variance[1], self.posterior_variance[1:])
                )
    
    def sample_t_with_weights(self,b_size,device):
        p = self.weights / np.sum(self.weights)
        indices_np = np.random.choice(len(p),size=b_size,p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / len(p) * p[indices_np]
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices,weights
    
    def predict_x_0_from_eps(self, x_t, t, eps):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape, x_t.device) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, x_t.device) * eps)
    
    def predict_eps_from_x_0(self, x_t, t, pred_x_0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape, x_t.device) * x_t
                - pred_x_0) \
               / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, x_t.device)
    
    def q_mean_variance(self, x_0, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
                extract(self.sqrt_alphas_cumprod, t, x_0.shape, x_0.device) * x_0
        )
        variance = extract(1.0 - self.alphas_cumprod, t, x_0.shape, x_0.device)
        log_variance = extract(
                self.log_one_minus_alphas_cumprod, t, x_0.shape, x_0.device
                )
        return mean, variance, log_variance
    
    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """

        # mu (x_t,x_0) = \frac{\sqrt{alphacumprod prev} betas}{1-alphacumprod} *x_0
        # + \frac{\sqrt{alphas}(1-alphacumprod prev)}{ 1- alphacumprod} * x_t
        posterior_mean = (extract(self.posterior_mean_coef1, t, x_t.shape, x_t.device) * x_0
                          + extract(self.posterior_mean_coef2, t, x_t.shape, x_t.device) * x_t)

        # var = \frac{1-alphacumprod prev}{1-alphacumprod} * betas
        posterior_var = extract(self.posterior_variance, t, x_t.shape, x_t.device)
        posterior_log_var_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape, x_t.device)
        return posterior_mean, posterior_var, posterior_log_var_clipped

    def p_mean_variance(self, model, x_t, t, estimate_noise=None):
        """
        Finds the mean & variance from N(x_{t-1}; mu_theta(x_t,t), sigma_theta (x_t,t))
        """
        if estimate_noise == None:
            estimate_noise = model(x_t, t)

        # fixed model variance defined as \hat{\beta}_t - could add learned parameter
        model_var = np.append(self.posterior_variance[1], self.betas[1:])
        model_logvar = np.log(model_var)
        model_var = extract(model_var, t, x_t.shape, x_t.device)
        model_logvar = extract(model_logvar, t, x_t.shape, x_t.device)

        pred_x_0 = self.predict_x_0_from_eps(x_t, t, estimate_noise).clamp(-1, 1)
        model_mean, _, _ = self.q_posterior_mean_variance(
                pred_x_0, x_t, t
                )
        return {
            "mean":         model_mean,
            "variance":     model_var,
            "log_variance": model_logvar,
            "pred_x_0":     pred_x_0,
            }

    def sample_p(self, model, x_t, t, denoise_fn="gauss"):
        out = self.p_mean_variance(model, x_t, t)
        # noise = torch.randn_like(x_t)
        if type(denoise_fn) == str:
            if denoise_fn == "gauss":
                noise = torch.randn_like(x_t)
            elif denoise_fn == "noise_fn":
                noise = self.noise_fn(x_t, t).float()
            elif denoise_fn == "random":
                # noise = random_noise(self.simplex, x_t, t).float()
                noise = torch.randn_like(x_t)
            else:
                noise = generate_simplex_noise(self.simplex, x_t, t, False, in_channels=self.img_channels).float()
        else:
            noise = denoise_fn(x_t, t)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        )
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_x_0": out["pred_x_0"]}
    
    def forward_backward(
            self, model, x, see_whole_sequence="half", t_distance=None, denoise_fn="gauss",
            ):
        assert see_whole_sequence == "whole" or see_whole_sequence == "half" or see_whole_sequence == None

        if t_distance == 0:
            return x.detach()

        if t_distance is None:
            t_distance = self.num_timesteps
        seq = [x.cpu().detach()]
        if see_whole_sequence == "whole":

            for t in range(int(t_distance)):
                t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
                # noise = torch.randn_like(x)
                noise = self.noise_fn(x, t_batch).float()
                with torch.no_grad():
                    x = self.sample_q_gradual(x, t_batch, noise)

                seq.append(x.cpu().detach())
        else:
            # x = self.sample_q(x,torch.tensor([t_distance], device=x.device).repeat(x.shape[0]),torch.randn_like(x))
            t_tensor = torch.tensor([t_distance - 1], device=x.device).repeat(x.shape[0])
            x = self.sample_q(
                    x, t_tensor,
                    self.noise_fn(x, t_tensor).float()
                    )
            if see_whole_sequence == "half":
                seq.append(x.cpu().detach())

        for t in range(int(t_distance) - 1, -1, -1):
            t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
            with torch.no_grad():
                out = self.sample_p(model, x, t_batch, denoise_fn)
                x = out["sample"]
            if see_whole_sequence:
                seq.append(x.cpu().detach())

        return x.detach() if not see_whole_sequence else seq
   
    def sample_q(self, x_0, t, noise):
        """
            q (x_t | x_0 )
        """
        return (extract(self.sqrt_alphas_cumprod, t, x_0.shape, x_0.device) * x_0 +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape, x_0.device) * noise)

    def sample_q_gradual(self, x_t, t, noise):
        """
        q (x_t | x_{t-1})
        """
        return (extract(self.sqrt_alphas, t, x_t.shape, x_t.device) * x_t +
                extract(self.sqrt_betas, t, x_t.shape, x_t.device) * noise)

    def calc_vlb_xt(self, model, x_0, x_t, t, estimate_noise=None):
        # find KL divergence at t
        true_mean, _, true_log_var = self.q_posterior_mean_variance(x_0, x_t, t)
        output = self.p_mean_variance(model, x_t, t, estimate_noise)
        kl = normal_kl(true_mean, true_log_var, output["mean"], output["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)
       

        decoder_nll = -discretised_gaussian_log_likelihood(
                x_0, output["mean"], log_scales=0.5 * output["log_variance"]
                )
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        nll = torch.where((t == 0), decoder_nll, kl)
        return {"output": nll, "pred_x_0": output["pred_x_0"]}

    def calc_loss(self, model, x_0, t):
        noise = torch.randn_like(x_0)
        #noise = self.noise_fn(x_0, t).float()

        x_t = self.sample_q(x_0, t, noise)
        
        estimate_noise = model(x_t, t)
        
        loss = {}
        if self.loss_type == "l1":
            loss["loss"] = mean_flat((estimate_noise - noise).abs())
        elif self.loss_type == "l2":
            loss["loss"] = mean_flat((estimate_noise - noise).square())
        elif self.loss_type == "hybrid":
            # add vlb term
            loss["vlb"] = self.calc_vlb_xt(model, x_0, x_t, t, estimate_noise)["output"]
            loss["loss"] = loss["vlb"] + mean_flat((estimate_noise - noise).square())
        else:
            loss["loss"] = mean_flat((estimate_noise - noise).square())
        return loss, x_t, estimate_noise

    def p_loss(self, model, x_0):
        if self.loss_weight == "none":
            if True:
                t = torch.randint(
                        0, min(800, self.num_timesteps), (x_0.shape[0],),
                        device=x_0.device
                        )
            else:
                t = torch.randint(0, self.num_timesteps, (x_0.shape[0],), device=x_0.device)
            weights = 1
        else:
            t, weights = self.sample_t_with_weights(x_0.shape[0], x_0.device)

        loss, x_t, eps_t = self.calc_loss(model, x_0, t)
        loss = ((loss["loss"] * weights).mean(), (loss, x_t, eps_t))
        return loss

    def prior_vlb(self, x_0):
        batch_size=1
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_0.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_0, t)
        kl_prior = normal_kl(
                mean1=qt_mean, logvar1=qt_log_variance, mean2=torch.tensor(0.0, device=x_0.device),
                logvar2=torch.tensor(0.0, device=x_0.device)
                )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_total_vlb(self, x_0, model):
        vb = []
        x_0_mse = []
        mse = []
        batch_size=1
        for t in reversed(list(range(self.num_timesteps))):
            t_batch = torch.tensor([t] * batch_size, device=x_0.device)
            noise = torch.randn_like(x_0)
            x_t = self.sample_q(x_0=x_0, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
          
            with torch.no_grad():
                out = self.calc_vlb_xt(
                        model,
                        x_0=x_0,
                        x_t=x_t,
                        t=t_batch,
                        )
                #print(out)
            vb.append(out["output"])
            x_0_mse.append(mean_flat((out["pred_x_0"] - x_0) ** 2))
            eps = self.predict_eps_from_x_0(x_t, t_batch, out["pred_x_0"])
            mse.append(mean_flat((eps - noise) ** 2))

      
        vb = torch.stack(vb, dim=1)
        x_0_mse = torch.stack(x_0_mse, dim=1)
        mse = torch.stack(mse, dim=1)

        prior_vlb = self.prior_vlb(x_0)
        total_vlb = vb.sum(dim=1) + prior_vlb
        
        return {
            "total_vlb": total_vlb,
            "prior_vlb": prior_vlb,
            "vb":        vb,
            "x_0_mse":   x_0_mse,
            "mse":       mse,
            }

    def detection_A(self, model, x_0, file, mask, total_avg=2):
        ARG_NUM = 1
        IMG_SIZE=256
        T=1000
        
        for i in [f"./diffusion-videos/ARGS={ARG_NUM}/Anomalous/{file[0]}",
                  f"./diffusion-videos/ARGS={ARG_NUM}/Anomalous/{file[0]}/{file[1]}/",
                  f"./diffusion-videos/ARGS={ARG_NUM}/Anomalous/{file[0]}/{file[1]}/A"]:
            try:
                os.makedirs(i)
            except OSError:
                pass

        for i in range(7, 0, -1):
            freq = 2 ** i
            self.noise_fn = lambda x, t: generate_simplex_noise(
                    self.simplex, x, t, False, frequency=freq,
                    in_channels=self.img_channels
                    )

            for t_distance in range(50, int(T * 0.6), 50):
                output = torch.empty((total_avg, 1,IMG_SIZE), device=x_0.device)
                for avg in range(total_avg):

                    t_tensor = torch.tensor([t_distance], device=x_0.device).repeat(x_0.shape[0])
                    x = self.sample_q(
                            x_0, t_tensor,
                            self.noise_fn(x_0, t_tensor).float()
                            )

                    for t in range(int(t_distance) - 1, -1, -1):
                        t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
                        with torch.no_grad():
                            out = self.sample_p(model, x, t_batch)
                            x = out["sample"]

                    output[avg, ...] = x

                # save image containing initial, each final denoised image, mean & mse
                output_mean = torch.mean(output, dim=0).reshape(1, 1, IMG_SIZE)
                mse = ((output_mean - x_0).square() * 2) - 1
                mse_threshold = mse > 0
                mse_threshold = (mse_threshold.float() * 2) - 1
                out = torch.cat([x_0, output[:3], output_mean, mse, mse_threshold, mask])

                temp = os.listdir(f'./diffusion-videos/ARGS={ARG_NUM}/Anomalous/{file[0]}/{file[1]}/A')

                plt.imshow(gridify_output(out, 4), cmap='gray')
                plt.axis('off')
                plt.savefig(
                        f'./diffusion-videos/ARGS={ARG_NUM}/Anomalous/{file[0]}/{file[1]}/A/freq={i}-t'
                        f'={t_distance}-{len(temp) + 1}.png'
                        )
                plt.clf()

    def detection_B(self, model, x_0, file, mask, denoise_fn="gauss", total_avg=5):
        assert type(file) == tuple
        ARG_NUM = 1
        IMG_SIZE=256
        T=1000
        for i in [f"./diffusion-videos/ARGS={ARG_NUM}/Anomalous/{file[0]}",
                  f"./diffusion-videos/ARGS={ARG_NUM}/Anomalous/{file[0]}/{file[1]}",
                  f"./diffusion-videos/ARGS={ARG_NUM}/Anomalous/{file[0]}/{file[1]}/{denoise_fn}"]:
            try:
                os.makedirs(i)
            except OSError:
                pass
        if denoise_fn == "octave":
            end = int(T * 0.6)
            self.noise_fn = lambda x, t: generate_simplex_noise(
                    self.simplex, x, t, False, frequency=64, octave=6,
                    persistence=0.8
                    ).float()
        else:
            end = int(T * 0.8)
            self.noise_fn = lambda x, t: torch.randn_like(x)

        dice_coeff = []
        for t_distance in range(50, end, 50):
            output = torch.empty((total_avg, 1, IMG_SIZE), device=x_0.device)
            for avg in range(total_avg):

                t_tensor = torch.tensor([t_distance], device=x_0.device).repeat(x_0.shape[0])
                x = self.sample_q(
                        x_0, t_tensor,
                        self.noise_fn(x_0, t_tensor).float()
                        )

                for t in range(int(t_distance) - 1, -1, -1):
                    t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
                    with torch.no_grad():
                        out = self.sample_p(model, x, t_batch)
                        x = out["sample"]

                output[avg, ...] = x

            # save image containing initial, each final denoised image, mean & mse
            output_mean = torch.mean(output, dim=[0]).reshape(1, 1, IMG_SIZE)

            temp = os.listdir(f'./diffusion-videos/ARGS={ARG_NUM}/Anomalous/{file[0]}/{file[1]}/{denoise_fn}')

            dice = evaluation.heatmap(
                    real=x_0, recon=output_mean, mask=mask,
                    filename=f'./diffusion-videos/ARGS={ARG_NUM}/Anomalous/{file[0]}/{file[1]}/'
                             f'{denoise_fn}/heatmap-t={t_distance}-{len(temp) + 1}.png'
                    )

            mse = ((output_mean - x_0).square() * 2) - 1
            mse_threshold = mse > 0
            mse_threshold = (mse_threshold.float() * 2) - 1
            out = torch.cat([x_0, output[:3], output_mean, mse, mse_threshold, mask])

            plt.imshow(gridify_output(out, 4), cmap='gray')
            plt.axis('off')
            plt.savefig(
                    f'./diffusion-videos/ARGS={ARG_NUM}/Anomalous/{file[0]}/{file[1]}/{denoise_fn}/t'
                    f'={t_distance}-{len(temp) + 1}.png'
                    )
            plt.clf()

            dice_coeff.append(dice)
        return dice_coeff

    def detection_A_fixedT(self, model, x_0, mask, end_freq=6):
        t_distance = 250
        IMG_SIZE=256
        
        output = torch.empty((6 * end_freq, 1,IMG_SIZE), device=x_0.device)
        for i in range(1, end_freq + 1):

            freq = 2 ** i
            noise_fn = lambda x, t: generate_simplex_noise(self.simplex, x, t, False, frequency=freq).float()

            t_tensor = torch.tensor([t_distance - 1], device=x_0.device).repeat(x_0.shape[0])
            x = self.sample_q(
                    x_0, t_tensor,
                    noise_fn(x_0, t_tensor).float()
                    )
            x_noised = x.clone().detach()
            for t in range(int(t_distance) - 1, -1, -1):
                t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
                with torch.no_grad():
                    out = self.sample_p(model, x, t_batch, denoise_fn=noise_fn)
                    x = out["sample"]

            mse = ((x_0 - x).square() * 2) - 1
            mse_threshold = mse > 0
            mse_threshold = (mse_threshold.float() * 2) - 1

            output[(i - 1) * 6:i * 6, ...] = torch.cat((x_0, x_noised, x, mse, mse_threshold, mask))

        return output