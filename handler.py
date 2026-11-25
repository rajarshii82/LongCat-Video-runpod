import os
import base64
import tempfile
import time
import traceback
import datetime

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

from longcat_video.modules.autoencoder_kl_wan import (
    AutoencoderKLWan
)

from longcat_video.modules.longcat_video_dit import (
    LongCatVideoTransformer3DModel
)

from longcat_video.context_parallel import context_parallel_util

from longcat_video.context_parallel.context_parallel_util import (
    init_context_parallel
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = os.getenv(
    "CHECKPOINT_DIR",
    "/runpod-volume/weights/LongCat-Video"
)

DEVICE = "cuda:0"
DTYPE = torch.bfloat16

pipe = None

REQUIRED_FOLDERS = [
    "tokenizer",
    "text_encoder",
    "vae",
    "scheduler",
    "dit",
    "lora"
]


# --------------------------------------------------
# Model storage validation
# --------------------------------------------------

def validate_weights():

    missing = []

    for folder in REQUIRED_FOLDERS:

        path = os.path.join(MODEL_PATH, folder)

        if not os.path.isdir(path):
            missing.append(folder)

    lora_path = os.path.join(
        MODEL_PATH,
        "lora/cfg_step_lora.safetensors"
    )

    if not os.path.isfile(lora_path):
        missing.append(
            "lora/cfg_step_lora.safetensors"
        )

    return missing


# --------------------------------------------------
# GPU diagnostics
# --------------------------------------------------

def gpu_info():

    available = torch.cuda.is_available()

    result = {
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cuda_available": available
    }

    if not available:
        return result

    props = torch.cuda.get_device_properties(0)

    result.update({
        "gpu_name": props.name,
        "gpu_architecture": list(
            torch.cuda.get_device_capability(0)
        ),
        "supported_architectures": (
            torch.cuda.get_arch_list()
        ),
        "gpu_memory_gb": round(
            props.total_memory / 1024**3, 2
        ),
        "allocated_memory_gb": round(
            torch.cuda.memory_allocated(0) / 1024**3, 2
        ),
        "reserved_memory_gb": round(
            torch.cuda.memory_reserved(0) / 1024**3, 2
        )
    })

    return result


# --------------------------------------------------
# Health check
# --------------------------------------------------

def health_check():

    missing_weights = validate_weights()

    result = {
        "status": "ok",
        **gpu_info(),
        "model_path": MODEL_PATH,
        "volume_mounted": os.path.ismount(
            "/runpod-volume"
        ),
        "model_directory_exists": os.path.isdir(
            MODEL_PATH
        ),
        "weights_found": not missing_weights,
        "missing_weights": missing_weights,
        "model_loaded": pipe is not None
    }

    # Verify that CUDA can actually execute an operation.

    if torch.cuda.is_available():

        try:

            test = torch.ones(
                (16, 16),
                device=DEVICE,
                dtype=DTYPE
            )

            output = test @ test

            torch.cuda.synchronize()

            result["cuda_test"] = "passed"

            del test
            del output

        except Exception as exc:

            result["cuda_test"] = "failed"
            result["cuda_error"] = str(exc)
            result["status"] = "degraded"

    else:
        result["status"] = "degraded"

    return result


# --------------------------------------------------
# Load LongCat once per worker
# --------------------------------------------------

def load_model():

    global pipe

    if pipe is not None:

        print(
            "Reusing previously loaded LongCat model",
            flush=True
        )

        return pipe

    start_time = time.perf_counter()

    print(
        f"Loading LongCat from: {MODEL_PATH}",
        flush=True
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is not available"
        )

    missing = validate_weights()

    if missing:

        raise RuntimeError(
            f"Missing model files or folders: {missing}"
        )

    print(
        f"GPU: {torch.cuda.get_device_name(0)}",
        flush=True
    )

    # Configure distributed execution for one GPU.

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    torch.cuda.set_device(0)

    if not dist.is_initialized():

        dist.init_process_group(
            backend="nccl",
            rank=0,
            world_size=1,
            timeout=datetime.timedelta(minutes=60)
        )

        init_context_parallel(
            context_parallel_size=1,
            global_rank=0,
            world_size=1
        )

    cp_size = context_parallel_util.get_cp_size()

    cp_split = context_parallel_util.get_optimal_split(
        cp_size
    )

    # Component loading helper.

    def load(component, folder, **kwargs):

        print(
            f"Loading component: {folder}",
            flush=True
        )

        return component.from_pretrained(
            MODEL_PATH,
            subfolder=folder,
            torch_dtype=DTYPE,
            **kwargs
        )

    print("Initializing LongCat pipeline", flush=True)

    new_pipe = LongCatVideoPipeline(

        tokenizer=AutoTokenizer.from_pretrained(
            MODEL_PATH,
            subfolder="tokenizer"
        ),

        text_encoder=load(
            UMT5EncoderModel,
            "text_encoder"
        ),

        vae=load(
            AutoencoderKLWan,
            "vae"
        ),

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

    print("Moving model to GPU", flush=True)

    new_pipe.to(0)

    # Load distilled inference LoRA.

    lora_path = os.path.join(
        MODEL_PATH,
        "lora/cfg_step_lora.safetensors"
    )

    print(
        f"Loading distilled LoRA: {lora_path}",
        flush=True
    )

    new_pipe.dit.load_lora(
        lora_path,
        "cfg_step_lora"
    )

    new_pipe.dit.enable_loras(
        ["cfg_step_lora"]
    )

    # Cache only after successful initialization.

    pipe = new_pipe

    torch.cuda.empty_cache()

    elapsed = time.perf_counter() - start_time

    print(
        f"Model loaded in {elapsed:.2f} seconds",
        flush=True
    )

    print(gpu_info(), flush=True)

    return pipe


# --------------------------------------------------
# Text-to-video generation
# --------------------------------------------------

def generate_video(data):

    prompt = data.get("prompt")

    if not isinstance(prompt, str) or not prompt.strip():

        raise ValueError(
            "A non-empty prompt is required"
        )

    seed = int(data.get("seed", 42))

    height = int(data.get("height", 480))
    width = int(data.get("width", 832))

    num_frames = int(
        data.get("num_frames", 93)
    )

    num_inference_steps = int(
        data.get("num_inference_steps", 16)
    )

    fps = 15

    # Conservative limits for initial deployment.

    if height <= 0 or width <= 0:

        raise ValueError(
            "Invalid video dimensions"
        )

    if height % 32 != 0 or width % 32 != 0:

        raise ValueError(
            "Height and width must be divisible by 32"
        )

    if height * width > 480 * 832:

        raise ValueError(
            "Initial worker supports up to 480p pixel count"
        )

    if num_frames < 5 or num_frames > 93:

        raise ValueError(
            "num_frames must be between 5 and 93"
        )

    if (num_frames - 1) % 4 != 0:

        raise ValueError(
            "num_frames must follow the 4n+1 format"
        )

    if not 1 <= num_inference_steps <= 50:

        raise ValueError(
            "num_inference_steps must be between 1 and 50"
        )

    total_start = time.perf_counter()

    pipeline = load_model()

    generator = torch.Generator(
        device=DEVICE
    )

    generator.manual_seed(seed)

    print(
        f"Generating video: "
        f"{width}x{height}, "
        f"{num_frames} frames, "
        f"{num_inference_steps} steps, "
        f"seed={seed}",
        flush=True
    )

    torch.cuda.reset_peak_memory_stats(0)

    inference_start = time.perf_counter()

    with torch.inference_mode():

        output = pipeline.generate_t2v(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            use_distill=True,
            guidance_scale=1.0,
            generator=generator
        )[0]

    torch.cuda.synchronize()

    inference_seconds = (
        time.perf_counter() - inference_start
    )

    peak_memory = round(
        torch.cuda.max_memory_allocated(0)
        / 1024**3,
        2
    )

    print(
        f"Inference completed in "
        f"{inference_seconds:.2f} seconds",
        flush=True
    )

    # Convert output to video frames.

    frames = torch.from_numpy(
        np.array(output)
    )

    frames = (
        (frames * 255)
        .clamp(0, 255)
        .to(torch.uint8)
    )

    del output

    actual_frames = int(frames.shape[0])

    # Encode MP4.

    with tempfile.TemporaryDirectory() as directory:

        path = os.path.join(
            directory,
            "output.mp4"
        )

        write_video(
            path,
            frames,
            fps=fps,
            video_codec="libx264",
            options={
                "crf": "18"
            }
        )

        file_size = os.path.getsize(path)

        with open(path, "rb") as file:

            video = base64.b64encode(
                file.read()
            ).decode("utf-8")

    del frames

    torch.cuda.empty_cache()

    total_seconds = (
        time.perf_counter() - total_start
    )

    return {
        "status": "success",
        "video_base64": video,
        "format": "mp4",
        "width": width,
        "height": height,
        "num_frames": actual_frames,
        "fps": fps,
        "duration_seconds": round(
            actual_frames / fps, 2
        ),
        "seed": seed,
        "num_inference_steps": num_inference_steps,
        "inference_seconds": round(
            inference_seconds, 2
        ),
        "total_seconds": round(
            total_seconds, 2
        ),
        "peak_allocated_gpu_memory_gb": peak_memory,
        "file_size_mb": round(
            file_size / 1024**2, 2
        )
    }


# --------------------------------------------------
# RunPod API handler
# --------------------------------------------------

def handler(job):

    try:

        data = job.get("input", {})

        if not isinstance(data, dict):

            raise ValueError(
                "input must be a JSON object"
            )

        action = data.get(
            "action",
            "text_to_video"
        )

        print(
            f"Received action: {action}",
            flush=True
        )

        if action == "health":

            return health_check()

        if action == "load_model":

            load_model()

            return {
                "status": "success",
                "model_loaded": True,
                **gpu_info()
            }

        if action == "text_to_video":

            return generate_video(data)

        return {
            "error": f"Unsupported action: {action}"
        }

    except Exception as exc:

        traceback.print_exc()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        return {
            "error": str(exc),
            "error_type": type(exc).__name__
        }


# --------------------------------------------------
# Worker entry point
# --------------------------------------------------

if __name__ == "__main__":

    print(
        "Starting LongCat RunPod Serverless Worker",
        flush=True
    )

    print(
        f"Checkpoint directory: {MODEL_PATH}",
        flush=True
    )

    print(
        f"PyTorch version: {torch.__version__}",
        flush=True
    )

    print(
        f"CUDA version: {torch.version.cuda}",
        flush=True
    )

    runpod.serverless.start({"handler": handler})