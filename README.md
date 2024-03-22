# OpenDMD

[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/collections/aaronb/dmd-65f92b47c8f264ce4de3f043) [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/aaronb/OpenDMD) 

Open source implementation and models of One-step Diffusion with Distribution Matching Distillation


| ![image (2)](https://github.com/Zeqiang-Lai/OpenDMD/assets/26198430/eae4aaef-4da8-4437-b544-2293d164a4cb) | ![image1](https://github.com/Zeqiang-Lai/OpenDMD/assets/26198430/5ae00440-6824-40c1-8132-d2b665c5844c) | ![image3](https://github.com/Zeqiang-Lai/OpenDMD/assets/26198430/b7e8509b-96d5-4526-961c-bf38b53c70bc) |  ![image2](https://github.com/Zeqiang-Lai/OpenDMD/assets/26198430/fe9c841a-9d17-49ae-a526-46ea5568b5e6) | 
| --- | --- | --- | --- | 

## Usage

Launch the gradio demo via:

```
python gradio_dmd.py
```

## Model Release

These models are very experimental releases, all of them are only trained with very few steps, so the performance is not satisfactory.

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
