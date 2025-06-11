# size of black-forest-labs/FLUX.1-dev: 24GB
# stabilityai/stable-diffusion-xl-base-1.0: 14GB
# stable-diffusion-v1-5/stable-diffusion-inpainting: 4.3GB

# import torch
# from diffusers import FluxPipeline

# def text_to_image(text: str):
#   pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
#   pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

#   image = pipe(
#     text,
#     height=1024,
#     width=1024,
#     guidance_scale=3.5,
#     num_inference_steps=50,
#     max_sequence_length=512,
#     generator=torch.Generator("cpu").manual_seed(0)
#   ).images[0]

#   return image

import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from utils.device import get_device_type, get_torch_dtype

def get_pipeline_embeds(pipeline: DiffusionPipeline, prompt: str, negative_prompt: str, device: str):
    """Get pipeline embeds for prompts bigger than the maxlength of the pipe
    :param pipeline:
    :param prompt:
    :param negative_prompt:
    :param device:
    :return:
    """
    max_length = pipeline.tokenizer.model_max_length

    # simple way to determine length of tokens
    input_ids = pipeline.tokenizer(
        prompt, return_tensors="pt", truncation=False
    ).input_ids.to(device)
    negative_ids = pipeline.tokenizer(
        negative_prompt, return_tensors="pt", truncation=False
    ).input_ids.to(device)

    # create the tensor based on which prompt is longer
    if input_ids.shape[-1] >= negative_ids.shape[-1]:
        shape_max_length = input_ids.shape[-1]
        negative_ids = pipeline.tokenizer(
            negative_prompt,
            truncation=False,
            padding="max_length",
            return_tensors="pt",
            max_length=shape_max_length,
        ).input_ids.to(device)

    else:
        shape_max_length = negative_ids.shape[-1]
        input_ids = pipeline.tokenizer(
            prompt,
            truncation=False,
            padding="max_length",
            return_tensors="pt",
            max_length=shape_max_length,
        ).input_ids.to(device)

    concat_embeds = []
    neg_embeds = []

    for i in range(0, shape_max_length, max_length):
        concat_embeds.append(pipeline.text_encoder(input_ids[:, i : i + max_length])[0])
        neg_embeds.append(pipeline.text_encoder(negative_ids[:, i : i + max_length])[0])

    return torch.cat(concat_embeds, dim=1), torch.cat(neg_embeds, dim=1)


def text_to_image(text: str):
    """
    Usually, a StableDiffusionPipeline will truncate input if it has more than 77 tokens.
    This method builds a stable diffusion pipeline which does not truncate input.
    """

    model_id = "sd-legacy/stable-diffusion-v1-5"
    # model_id = "crynux-ai/stable-diffusion-v1-5"

    device = get_device_type()
    dtype = get_torch_dtype(device)

    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        # use_safetensors=True,
    )

    # only for mps, without the following line, mps will be slower than cpu.
    torch.device(device.value)

    pipeline = pipeline.to(device.value)

    prompt = text
    negative_prompt = ""

    prompt_embeds, negative_prompt_embeds = get_pipeline_embeds(
        pipeline, prompt, negative_prompt, device.value
    )

    # Forward
    image = pipeline(
        prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds
    ).images[0]

    return image


# Token indices sequence length is longer than the specified maximum sequence length for this model (? > 77). Running this sequence through the model will result in indexing errors
def text_to_image_truncated(text: str):
    model_id = "sd-legacy/stable-diffusion-v1-5"

    device: str
    pipe: StableDiffusionPipeline

    if torch.cuda.is_available():
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            # use_safetensors=True,
        )

        device = "cuda"

    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_id)

        device = "cpu"

    pipe = pipe.to(device)

    image = pipe(text).images[0]

    return image


# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()
