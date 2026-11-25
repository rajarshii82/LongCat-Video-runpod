import os
import base64
import tempfile

import runpod
import torch
import torch.distributed as dist
import numpy as np

from transformers import AutoTokenizer, UMT5EncoderModel
from torchvision.io import write_video

from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler
)
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import (
    LongCatVideoTransformer3DModel
)
from longcat_video.context_parallel import context_parallel_util
from longcat_video.context_parallel.context_parallel_util import (
    init_context_parallel
)

MODEL_PATH = os.getenv(
    "CHECKPOINT_DIR",
    "/runpod-volume/weights/LongCat-Video"
)

pipe = None


def load_model():
    global pipe

    if pipe is not None:
        return pipe

    print("Loading model from:", MODEL_PATH)

    if not os.path.isdir(MODEL_PATH):
        raise RuntimeError(
            f"Model directory not found: {MODEL_PATH}"
        )

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    torch.cuda.set_device(0)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=0,
            world_size=1
        )

        init_context_parallel(
            context_parallel_size=1,
            global_rank=0,
            world_size=1
        )

    cp_split = context_parallel_util.get_optimal_split(1)

    def load(component, folder, **kwargs):
        return component.from_pretrained(
            MODEL_PATH,
            subfolder=folder,
            torch_dtype=torch.bfloat16,
            **kwargs
        )

    pipe = LongCatVideoPipeline(
        tokenizer=AutoTokenizer.from_pretrained(
            MODEL_PATH,
            subfolder="tokenizer"
        ),
        text_encoder=load(
            UMT5EncoderModel, "text_encoder"
        ),
        vae=load(AutoencoderKLWan, "vae"),
        scheduler=load(
            FlowMatchEulerDiscreteScheduler,
            "scheduler"
        ),
        dit=load(
            LongCatVideoTransformer3DModel,
            "dit",
            cp_split_hw=cp_split
        )
    )

    pipe.to(0)

    pipe.dit.load_lora(
        os.path.join(
            MODEL_PATH,
            "lora/cfg_step_lora.safetensors"
        ),
        "cfg_step_lora"
    )

    pipe.dit.enable_loras(["cfg_step_lora"])

    print("Model loaded successfully")

    return pipe


def handler(job):
    data = job["input"]

    if data.get("action") == "health":
        return {
            "status": "ok",
            "cuda_available": torch.cuda.is_available(),
            "model_path": MODEL_PATH,
            "weights_found": os.path.isdir(MODEL_PATH),
            "model_loaded": pipe is not None
        }

    prompt = data["prompt"]
    seed = int(data.get("seed", 42))
    pipeline = load_model()

    generator = torch.Generator(device="cuda:0")
    generator.manual_seed(seed)

    with torch.inference_mode():
        output = pipeline.generate_t2v(
            prompt=prompt,
            height=480,
            width=832,
            num_frames=93,
            num_inference_steps=16,
            use_distill=True,
            guidance_scale=1.0,
            generator=generator
        )[0]

    frames = torch.from_numpy(np.array(output))
    frames = (frames * 255).clamp(0, 255).to(torch.uint8)

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "output.mp4")

        write_video(
            path,
            frames,
            fps=15,
            video_codec="libx264"
        )

        with open(path, "rb") as file:
            video = base64.b64encode(
                file.read()
            ).decode("utf-8")

    return {
        "video_base64": video,
        "format": "mp4",
        "seed": seed
    }


runpod.serverless.start({"handler": handler})