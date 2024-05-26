import argparse
import time

import gradio as gr
import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel

from dmd.scheduling_dmd import DMDScheduler

parser = argparse.ArgumentParser()
parser.add_argument("--unet-path", default='Lykon/dreamshaper-8')
parser.add_argument("--model-path", default='aaronb/dreamshaper-8-dmd-1kstep')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

unet = UNet2DConditionModel.from_pretrained(args.unet_path, subfolder='unet')
pipe = DiffusionPipeline.from_pretrained(args.model_path, unet=unet)
pipe.scheduler = DMDScheduler.from_config(pipe.scheduler.config)
pipe.to(device=device, dtype=torch.float16)


def predict(prompt, seed=1231231):
    generator = torch.manual_seed(seed)
    last_time = time.time()

    image = pipe(
        prompt,
        num_inference_steps=1,
        guidance_scale=0.0,
        generator=generator,
    ).images[0]

    print(f"Pipe took {time.time() - last_time} seconds")
    return image


css = """
#container{
    margin: 0 auto;
    max-width: 40rem;
}
#intro{
    max-width: 100%;
    text-align: center;
    margin: 0 auto;
}
"""
with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="container"):
        gr.Markdown(
            """# Distribution Matching Distillation
            """,
            elem_id="intro",
        )
        with gr.Row():
            with gr.Row():
                prompt = gr.Textbox(placeholder="Insert your prompt here:", scale=5, container=False)
                generate_bt = gr.Button("Generate", scale=1)

        image = gr.Image(type="filepath")
        with gr.Accordion("Advanced options", open=False):
            seed = gr.Slider(randomize=True, minimum=0, maximum=12013012031030, label="Seed", step=1)

        inputs = [prompt, seed]
        generate_bt.click(fn=predict, inputs=inputs, outputs=image, show_progress=False)
        prompt.input(fn=predict, inputs=inputs, outputs=image, show_progress=False)
        seed.change(fn=predict, inputs=inputs, outputs=image, show_progress=False)

demo.queue(api_open=False)
demo.launch(show_api=False)
