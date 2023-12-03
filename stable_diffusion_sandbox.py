from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
# pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16")
# pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")


import torch

from PIL import Image

import numpy as np

# from diffusers.utils import load_image

# init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png").resize((512, 512))

#%%
# prompt = "hey give me a painting that shows 0.8 of inward attention (which is alpha) and 0.5 of focus (theta) and 0.7 of being in sunchrony. awawa: "+str(np.random.randint(101212))
prompt = "hey give me a painting that shows 0.8 of inward attention (which is alpha) and 0.5 of focus (theta) and 0.7 of being in sunchrony."
# prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"

# generator = torch.Generator(device="cuda").manual_seed(42069)

# image = pipe(prompt=prompt, generator=generator, image=show_image, strength=0.0, num_inference_steps=1, guidance_scale=0.0).images[0]
image = pipe(prompt=prompt, image=image, generator=generator, strength=-1, num_inference_steps=1, guidance_scale=0.0).images[0]
# image = pipe(prompt=prompt, strength=0.01, num_inference_steps=3, guidance_scale=0.0).images[0]
image

# show_image = image
# show_image = Image.fromarray((np.asarray(show_image).astype(np.float32)*0.66 + 0.33*np.asarray(image).astype(np.float32)).astype(np.int8), 'RGB')
# show_image = Image.fromarray(np.asarray(image), 'RGB')
# show_image
