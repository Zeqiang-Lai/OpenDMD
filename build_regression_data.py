import argparse
import multiprocessing as mp
import os
import time
import signal
import json
import random

import torch
from diffusers import AutoPipelineForText2Image, DEISMultistepScheduler


def run(device_id, job_id, worker_id, n_gpu, n_worker, caption_path, model_id, save_dir, size=None):
    global_id = device_id * n_worker + worker_id
    n_job = n_gpu * n_worker
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    print(f"[{job_id}] Using device: {device_id} global_id {global_id}")

    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, local_files_only=True)
    pipe.safety_checker = None
    pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(f"cuda")
    pipe.set_progress_bar_config(disable=True)

    # load captions part
    with open(caption_path, "r") as f:
        prompts = f.readlines()
        if size is not None:
            prompts = random.choices(prompts, k=size)

    chunk_size = len(prompts) // n_job
    prompts = prompts[global_id * chunk_size : (global_id + 1) * chunk_size]
    print("[{}] process chunk size {}, range [{}:{}]".format(job_id, chunk_size, global_id * chunk_size, (global_id + 1) * chunk_size))

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"meta_{job_id}.json"), "a") as f:
        for i, prompt in enumerate(prompts):
            prompt = prompt.strip()

            batch_size = 1
            height = 512
            width = 512
            num_inference_steps = 25
            try:
                num_channels_latents = pipe.unet.config.in_channels
            except Exception as e:
                num_channels_latents = pipe.transformer.config.in_channels

            latents = pipe.prepare_latents(
                batch_size,
                num_channels_latents,
                height,
                width,
                dtype=torch.float16,
                device=pipe.device,
                generator=torch.Generator().manual_seed(i),
            )

            image = pipe(prompt=prompt, latents=latents, num_inference_steps=num_inference_steps).images[0]

            image_path = os.path.join("images", f"{job_id}", f"{i}.jpg")
            latent_path = os.path.join("latents", f"{job_id}", f"{i}.pt")

            os.makedirs(os.path.dirname(os.path.join(save_dir, image_path)), exist_ok=True)
            os.makedirs(os.path.dirname(os.path.join(save_dir, latent_path)), exist_ok=True)

            image.save(os.path.join(save_dir, image_path))
            torch.save(latents, os.path.join(save_dir, latent_path))

            f.write(json.dumps({"image_path": image_path, "latent_path": latent_path, "prompt": prompt, "seed": i}) + "\n")
            f.flush()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default=None, type=str, help="Comma separated list of GPUs to use")
    parser.add_argument("--workers", default=1, type=int, help="Number of workers spawned per GPU (default 1)")
    parser.add_argument("--size", default=None, type=int)
    parser.add_argument("--caption_path", default="diffusion_db_prompts.txt", type=str)
    parser.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5", type=str)
    parser.add_argument("--save_dir", default="data/diffusion_db_runwayml_stable-diffusion-v1-5", type=str)
    args = parser.parse_args()
    return args


def main():
    mp.set_start_method("spawn")
    args = parse_args()

    num_gpus = torch.cuda.device_count()
    num_workers = args.workers
    if args.gpus is None:
        visible_gpus = list(range(num_gpus))
    else:
        visible_gpus = []
        parts = args.gpus.split(",")
        for p in parts:
            if "-" in p:
                lo, hi = p.split("-")
                lo, hi = int(lo), int(hi)
                assert hi >= lo
                visible_gpus.extend(list(range(lo, hi + 1)))
            else:
                visible_gpus.append(int(p))

    visible_gpus = list(set(visible_gpus))  # keep distinct
    assert len(visible_gpus) > 0

    print("Using GPUs: ", visible_gpus)

    kwargs = dict(
        device_id=device_id,
        job_id=job_id,
        worker_id=i,
        n_gpu=len(visible_gpus),
        n_worker=num_workers,
        caption_path=args.caption_path,
        model_id=args.model_id,
        save_dir=args.save_dir,
        size=args.size,
    )

    jobs = {}
    for device_id in visible_gpus:
        for i in range(num_workers):
            job_id = f"GPU{device_id:02d}-{i}"
            print(f"[{job_id}] Launching worker-process...")
            p = mp.Process(target=run, kwargs=kwargs)
            jobs[job_id] = (p, device_id)
            p.start()

    try:
        while True:
            time.sleep(1)
            for job_id, (job, device_id) in jobs.items():
                if job.is_alive():
                    pass
                else:
                    print(f"[{job_id}] Worker died, cleaning up...")
                    # remove remaining tar file
                    os.system(f"rm -r -f -v {args.save_dir}/images/{job_id}")
                    os.system(f"rm -r -f -v {args.save_dir}/latents/{job_id}")
                    os.system(f"rm -f -v {args.save_dir}/meta_{job_id}.json")
                    os.system(f"rm -f -v {args.save_dir}/meta_{job_id}.json")
                    print(f"[{job_id}] respawning...")
                    p = mp.Process(target=run, kwargs=kwargs)
                    jobs[job_id] = (p, device_id)
                    p.start()

    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        for job_id, (job, device_id) in jobs.items():
            job.terminate()
        for job_id, (job, device_id) in jobs.items():
            print(f"[{job_id}] waiting for exit...")
            job.join()
        print("done.")


if __name__ == "__main__":
    main()
