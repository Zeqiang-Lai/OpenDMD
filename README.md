# OpenDMD

[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/collections/aaronb/dmd-65f92b47c8f264ce4de3f043) [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/aaronb/OpenDMD) 

Open source implementation and models of One-step Diffusion with Distribution Matching Distillation





| ![image (2)](https://github.com/Zeqiang-Lai/OpenDMD/assets/26198430/41b7c333-619a-48ed-baf8-45687516d53a) | ![image1](https://github.com/Zeqiang-Lai/OpenDMD/assets/26198430/0e092092-1835-4314-aaa3-c5f6ba50edbe) | ![image3](https://github.com/Zeqiang-Lai/OpenDMD/assets/26198430/80b86719-b647-4477-9e1c-d1b9ad14d9f3) | ![image2](https://github.com/Zeqiang-Lai/OpenDMD/assets/26198430/eb922a2f-dbd4-4607-92a7-a2eee8f1be4b) | 
| --- | --- | --- | --- | 

## Usage

Launch the gradio demo via:

```
python gradio_dmd.py
```

## Model Release

These models are very experimental releases, all of them are only trained with very few steps, so the performance is not satisfactory.

If you are trying to reproduce the results, we recommend first testing original sd1.5 (which is fully verified), there might be some issues for fintuned model.

| Model                            | Link                                                                       |
| -------------------------------- | -------------------------------------------------------------------------- |
| dreamshaper-8-dmd-1kstep         | [hf-model](https://huggingface.co/aaronb/dreamshaper-8-dmd-1kstep)         |
| dreamshaper-8-dmd-kl-only-6kstep | [hf-model](https://huggingface.co/aaronb/dreamshaper-8-dmd-kl-only-6kstep) |

## Real Time Demo

The video is not accelerated.

https://github.com/Zeqiang-Lai/OpenDMD/assets/26198430/f8069208-934d-49a2-839e-689c2e1a85fe

## Citation

```bibtex
@article{yin2023one,
  title={One-step Diffusion with Distribution Matching Distillation},
  author={Yin, Tianwei and Gharbi, Micha{\"e}l and Zhang, Richard and Shechtman, Eli and Durand, Fredo and Freeman, William T and Park, Taesung},
  journal={arXiv preprint arXiv:2311.18828},
  year={2023}
}
```
