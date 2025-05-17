import copy
import gc
import sys
import warnings

import requests
import torch
from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                             IMAGE_TOKEN_INDEX)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (get_model_name_from_path, process_images,
                            tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from PIL import Image
from io import BytesIO

class LLaVAOneVision:
    def __init__(self, model_path=None, do_sample=False, max_new_tokens=1024, top_k=10, top_p=0.001, temperature=0.1, repetition_penalty=1.0):
        assert model_path is not None, "Please provide model path"
        model_name = "llava_qwen"
        llava_model_args = {
            "multimodal": True,
            "attn_implementation": "sdpa",
        }
        overwrite_config = {}
        overwrite_config["image_aspect_ratio"] = "pad"
        llava_model_args["overwrite_config"] = overwrite_config
        self.tokenizer, self.model, self.image_processor, max_length = load_pretrained_model(model_path, None, model_name, device_map="auto", **llava_model_args)
        self.model.eval()

        if not do_sample:
            self.gen_config = {
                "do_sample": False,
                "max_new_tokens": max_new_tokens,
                "top_k": None,
                "top_p": None,
                "temperature": 0,
                "repetition_penalty": repetition_penalty
            }
        else:
            self.gen_config = {
                "do_sample": True,
                "max_new_tokens": max_new_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
            }

    def load_image(self, img_url):
        if img_url.startswith("http://") or img_url.startswith("https://"):
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.66 Safari/537.36',
            }
            response = requests.get(img_url, headers=headers).content
            image_io = BytesIO(response)
            image = Image.open(image_io)
        else:
            image = Image.open(img_url)
        return image
    
    def parse_input(self, query = None, imgs = None):
        inputs = {}

        if imgs is None:
            image_sizes = None
            image_tensor = None
        else:
            if isinstance(imgs, str):
                imgs = [imgs]
            
            imgs = [self.load_image(img) for img in imgs]
            image_sizes = [image.size for image in imgs]
            image_tensor = process_images(imgs, self.image_processor, self.model.config)
            image_tensor = [_image.to(dtype=torch.float16, device="cuda") for _image in image_tensor]
            
        question = ""
        for cur_i in range(len(imgs)):
            question += f"Image-{cur_i+1}: {DEFAULT_IMAGE_TOKEN}\n"
        question += f"{query}"
        # for _ in imgs:
        #     question = DEFAULT_IMAGE_TOKEN + question
        conv_template = "qwen_1_5"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda")
        inputs['input_ids'] = input_ids
        inputs['images'] = image_tensor
        inputs['image_sizes'] = image_sizes
        return inputs

    def infer(self, query=None, imgs=None):
        inputs = self.parse_input(query, imgs)
        input_ids = inputs.pop('input_ids')
        images = inputs.pop('images')
        image_sizes = inputs.pop('image_sizes')
        conversation = self.model.generate(
            input_ids, images=images, image_sizes=image_sizes,
            **self.gen_config
        )
        response = self.tokenizer.batch_decode(conversation, skip_special_tokens=True)
        del input_ids, images, image_sizes
        torch.cuda.empty_cache()
        gc.collect()
        return response[0]
