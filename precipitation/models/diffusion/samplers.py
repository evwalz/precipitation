import torch
from torch import nn

from precipitation.models.diffusion.beta_schedules import *


class DDPM_Sampler(nn.Module):
    
    def __init__(self,
                 num_timesteps=1000,
                 schedule='linear'
                ):
        
        super().__init__()
        
        self.num_timesteps=num_timesteps
        self.schedule=schedule
        
        self.register_buffer('betas',get_beta_schedule(self.schedule,self.num_timesteps))
        self.register_buffer('betas_sqrt',self.betas.sqrt())
        self.register_buffer('alphas',1-self.betas)
        self.register_buffer('alphas_cumprod',torch.cumprod(self.alphas,0))
        self.register_buffer('alphas_cumprod_sqrt',self.alphas_cumprod.sqrt())
        self.register_buffer('alphas_one_minus_cumprod_sqrt',(1-self.alphas_cumprod).sqrt())
        self.register_buffer('alphas_sqrt',self.alphas.sqrt())
        self.register_buffer('alphas_sqrt_recip',1/(self.alphas_sqrt))
        
    @torch.no_grad()
    def forward(self,*args,**kwargs):   
        return self.step(*args,**kwargs)
    
    @torch.no_grad()
    def step(self,x_t,t,z_t):
        """
            Given approximation of noise z_t in x_t predict x_(t-1)
        """
        assert (t<self.num_timesteps).all()
        
        # 2. Approximate Distribution of Previous Sample in the chain 
        mean_pred,std_pred=self.posterior_params(x_t,t,z_t)
        
        # 3. Sample from the distribution
        z=torch.randn_like(x_t) if any(t>0) else torch.zeros_like(x_t)
        return mean_pred + std_pred*z
    
    def posterior_params(self,x_t,t,noise_pred):
        
        assert (t<self.num_timesteps).all()
        
        beta_t=self.betas[t].view(x_t.shape[0],1,1,1)
        alpha_one_minus_cumprod_sqrt_t=self.alphas_one_minus_cumprod_sqrt[t].view(x_t.shape[0],1,1,1)
        alpha_sqrt_recip_t=self.alphas_sqrt_recip[t].view(x_t.shape[0],1,1,1)
        
        mean=alpha_sqrt_recip_t*(x_t-beta_t*noise_pred / alpha_one_minus_cumprod_sqrt_t)
        std=self.betas_sqrt[t].view(x_t.shape[0],1,1,1)
        
        return mean, std
    

class DDIM_Sampler(nn.Module):
    
    def __init__(self,
                 num_timesteps=100,
                 train_timesteps=1000,
                 clip_sample=True,
                 schedule='linear'
                ):
        
        super().__init__()
        
        self.num_timesteps=num_timesteps
        self.train_timesteps=train_timesteps
        self.ratio=self.train_timesteps // self.num_timesteps
        self.final_alpha_cumprod=torch.tensor([1.0])
        self.clip_sample=clip_sample
        self.schedule=schedule
        
        self.register_buffer('betas',get_beta_schedule(self.schedule,self.train_timesteps))
        self.register_buffer('betas_sqrt',self.betas.sqrt())
        self.register_buffer('alphas',1-self.betas)
        self.register_buffer('alphas_cumprod',torch.cumprod(self.alphas,0))
        self.register_buffer('alphas_cumprod_sqrt',self.alphas_cumprod.sqrt())
        self.register_buffer('alphas_one_minus_cumprod_sqrt',(1-self.alphas_cumprod).sqrt())
        self.register_buffer('alphas_sqrt',self.alphas.sqrt())
        self.register_buffer('alphas_sqrt_recip',1/(self.alphas_sqrt))
        
    @torch.no_grad()
    def forward(self,*args,**kwargs):   
        return self.step(*args,**kwargs)
    
    @torch.no_grad()
    def step(self,x_t,t,z_t,eta=0):
        """
            Given approximation of noise z_t in x_t predict x_(t-1)
        """
        assert (t<self.num_timesteps).all()
        b,c,h,w=z_t.shape
        device=z_t.device
        
        # compute steps based on the ratio (number of DDPM steps to take in each DDIM)
        t=t*self.ratio
        t_prev=t-self.ratio

        # parameters        
        alpha_cumprod_prev = self.alphas_cumprod[t_prev].where(t_prev.ge(0), self.final_alpha_cumprod.to(device)) # >= 0
        alpha_cumprod_prev = alpha_cumprod_prev.view(b,1,1,1)
        alpha_cumprod_prev_sqrt = self.alphas_cumprod_prev_sqrt[t_prev]
        
        # estimate origin
        x_0_pred=self.estimate_origin(x_t,t,z_t)
        if self.clip_sample:
            x_0_pred=torch.clamp(x_0_pred, -1, 1)
            
        std_dev_t = eta * self.estimate_std(t,t_prev).view(b,1,1,1)
        x_0_grad = (1-alpha_cumprod_prev - std_dev_t**2).sqrt() * z_t
        prev_sample=alpha_cumprod_prev_sqrt*x_0_pred + x_0_grad

        if eta > 0:          
            noise = torch.randn(model_output.shape, dtype=model_output.dtype)  # this might require fixing
            prev_sample = prev_sample+std_dev_t*eta*noise

        return prev_sample
    
    def estimate_std(self, t,t_prev):
        alpha_cumprod = self.alphas_cumprod[t]
        alpha_cumprod_prev = self.alphas_cumprod[t_prev].where(t_prev.gt(0), self.final_alpha_cumprod.to(alpha_cumprod.device))
        one_minus_alpha_cumprod = 1 - alpha_cumprod
        one_minus_alpha_cumprod_prev = 1 - alpha_cumprod_prev

        var = (one_minus_alpha_cumprod_prev / one_minus_alpha_cumprod) * (1 - alpha_cumprod / alpha_cumprod_prev)

        return var.sqrt()
    
    def estimate_origin(self,x_t,t,z_t):
        alpha_cumprod = self.alphas_cumprod[t].view(b,1,1,1)
        alpha_one_minus_cumprod_sqrt=self.alphas_one_minus_cumprod_sqrt[t]
        return(x_t-alpha_one_minus_cumprod_sqrt*z_t)/alpha_cumprod.sqrt()
