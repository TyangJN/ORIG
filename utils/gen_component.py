import os
from openai import OpenAI
import base64

import torch
import json
import PIL

from google import genai
from PIL import Image
from io import BytesIO
import os



OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
SERPER_API_KEY = "YOUR_SERPER_API_KEY"
SERPAPI_API_KEY = "YOUR_SERPAPI_API_KEY"
JINA_API_KEY = "YOUR_JINA_API_KEY"
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"


def gen_gpt(idx, prompt, gen_path, modality):
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    client = OpenAI()
    img_save_path = f'{gen_path}/{idx}_{modality}.png'

    if os.path.exists(img_save_path):
        print(img_save_path + " already exists")
        return

    if modality == "mm" or modality == "img":
        img_paths = []

        for img_path in prompt['img_path']:
            img_paths.append(open(f"{img_path}", 'rb'))

        result = client.images.edit(
            model="gpt-image-1",
            image=img_paths,
            prompt=prompt['prompt']
        )

        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)

        # Save the image to a file
        with open(img_save_path, "wb") as f:
            f.write(image_bytes)

    elif modality == "txt" or modality == "cot" or modality == "dir":
        result = client.images.generate(
            model="gpt-image-1",
            prompt=prompt['prompt']
        )
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)

        # Save the image to a file
        with open(img_save_path, "wb") as f:
            f.write(image_bytes)

    else:
        raise ValueError(f"Invalid modality: {modality}")
    return


def gen_gemini(idx, prompt, gen_path, modality):
    client = genai.Client(api_key=GEMINI_API_KEY)
    img_save_path = f'{gen_path}/{idx}_{modality}.png'

    if modality == "mm" or modality == "img":
        imgs = []
        for img_path in prompt['img_path']:
            imgs.append(PIL.Image.open(img_path))
        
        contents = [prompt['prompt']]
        contents.extend(imgs)

        response = client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=contents,
        )

        for part in response.candidates[0].content.parts:
            if part.text is not None:
                print(part.text)
            elif part.inline_data is not None:
                image = Image.open(BytesIO(part.inline_data.data))
                image.save(img_save_path)
        return

    elif modality == "txt" or modality == "cot" or modality == "dir":
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=[prompt['prompt']],
        )

        for part in response.candidates[0].content.parts:
            if part.text is not None:
                print(part.text)
            elif part.inline_data is not None:
                image = Image.open(BytesIO(part.inline_data.data))
                image.save(img_save_path)
        return


def gen_qwen(idx, prompt, gen_path, modality):
    from diffusers import DiffusionPipeline
    from diffusers import QwenImageEditPipeline

    img_save_path = f'{gen_path}/{idx}_{modality}.png'

    if modality == "mm" or modality == "img":
        pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
        print("pipeline loaded")
        pipeline.to(torch.bfloat16)
        pipeline.to("cuda")
        pipeline.set_progress_bar_config(disable=None)
        

        image = Image.open(prompt['img_path'][0]).convert("RGB")

        inputs = {
            "image": image,
            "prompt": prompt['prompt'],
            "generator": torch.manual_seed(0),
            "true_cfg_scale": 4.0,
            "negative_prompt": " ",
            "num_inference_steps": 50,
        }

        with torch.inference_mode():
            output = pipeline(**inputs)
            output_image = output.images[0]
            output_image.save(img_save_path)
        return

    elif modality == "txt" or modality == "cot" or modality == "dir":
        # Load the pipeline
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16
            device = "cuda"
        else:
            torch_dtype = torch.float32
            device = "cpu"

        model_name = "Qwen/Qwen-Image"
        pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
        pipe = pipe.to(device)  

        positive_magic = {
            "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
            "zh": ", 超清，4K，电影级构图." # for chinese prompt
        }

        # Generate image
        negative_prompt = " " # using an empty string if you do not have specific concept to remove

        # Generate with different aspect ratios
        aspect_ratios = {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1140),
            "3:4": (1140, 1472),
            "3:2": (1584, 1056),
            "2:3": (1056, 1584),
        }

        width, height = aspect_ratios["1:1"]

        image = pipe(
            prompt=prompt['prompt'] + positive_magic["en"],
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=50,
            true_cfg_scale=4.0,
            generator=torch.Generator(device="cuda").manual_seed(42)
        ).images[0]

        image.save(img_save_path)
        return

    else:
        raise ValueError(f"Invalid modality: {modality}")
    return


def gen_flux(prompt, gen_path, modality):
    if modality == "mm" or modality == "img":
        return
    elif modality == "txt" or modality == "cot" or modality == "dir":
        return
    else:
        raise ValueError(f"Invalid modality: {modality}")
    return


def gen_emu(prompt, gen_path, modality):
    if modality == "mm" or modality == "img":
        return
    elif modality == "txt" or modality == "cot" or modality == "dir":
        return
    else:
        raise ValueError(f"Invalid modality: {modality}")
    return


def gen_sd(prompt, gen_path, modality):
    if modality == "mm" or modality == "img":
        return
    elif modality == "txt" or modality == "cot" or modality == "dir":
        return
    else:
        raise ValueError(f"Invalid modality: {modality}")
    return
