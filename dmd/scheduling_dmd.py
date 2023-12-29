from dataclasses import dataclass
from typing import List, Tuple, Union, Optional

import torch
from diffusers import DDPMScheduler
from diffusers.utils import BaseOutput


@dataclass
class DMDSchedulerOutput(BaseOutput):
    pred_original_sample: Optional[torch.FloatTensor] = None


class DMDScheduler(DDPMScheduler):
    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[List[int]] = None,
    ):
        self.timesteps = torch.tensor([self.config.num_train_timesteps-1]).long().to(device)

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator=None,
        return_dict: bool = True,
    ) -> Union[DMDSchedulerOutput, Tuple]:
        t = self.config.num_train_timesteps - 1

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t

        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        if not return_dict:
            return (pred_original_sample,)

        return DMDSchedulerOutput(pred_original_sample=pred_original_sample)
