import math
import requests
import torch
import torchvision.transforms as T

from io import BytesIO
from PIL import Image
from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig, GenerationConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

class InternVL2:
    def __init__(self, model_path=None, cache_max_entry_count=0.8, system_prompt=None, context_max_len=8192, rope_scaling_factor=2.5, do_sample=False, max_new_tokens=1024, top_k=20, top_p=0.8, temperature=1.0, repetition_penalty=1.0):
        assert model_path is not None, "Please provide model path"

        if not do_sample:
            self.gen_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                top_k=1,
                temperature=0,
                repetition_penalty=repetition_penalty
            )
        else:
            self.gen_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty
            )

        # self.model = pipeline(model_path, chat_template_config=chat_template_config, backend_config=TurbomindEngineConfig(tp=torch.cuda.device_count(), session_len=context_max_len, rope_scaling_factor=rope_scaling_factor, cache_max_entry_count=cache_max_entry_count))
        if context_max_len == 8192:
            self.model = pipeline(
                model_path,
                backend_config=TurbomindEngineConfig(tp=torch.cuda.device_count(), session_len=context_max_len, cache_max_entry_count=cache_max_entry_count)
            )
        else:
            self.model = pipeline(
                model_path,
                backend_config=TurbomindEngineConfig(
                    tp=torch.cuda.device_count(),
                    session_len=context_max_len,
                    cache_max_entry_count=cache_max_entry_count,
                    rope_scaling_factor=rope_scaling_factor
                )
            )
    
    def load_image(self, img_url):
        return load_image(img_url)
    
    def parse_input(self, query=None, imgs=None):
        if imgs is None:
            prompt = query
            return prompt
        else:
            if isinstance(imgs, list) and len(imgs) == 1:
                imgs = imgs[0]

            if isinstance(imgs, list):
                images = [self.load_image(img_url) for img_url in imgs]
                prompt_prefix = ""
                for i in range(len(images)):
                    prompt_prefix += f'Image-{i+1}: {IMAGE_TOKEN}\n'
                prompt = prompt_prefix + query
            else:
                images = self.load_image(imgs)
                prompt = query
            return (prompt, images)

    def infer(self, query = None, imgs = None):
        inputs = self.parse_input(query, imgs)
        response = self.model(inputs, gen_config=self.gen_config).text
        return response

    def batch_infer(self, query_list = None, imgs_list = None):
        inputs_list = [self.parse_input(query, imgs) for query, imgs in zip(query_list, imgs_list)]
        response_list = self.model(inputs_list, gen_config=self.gen_config)
        response_list = [response.text for response in response_list]
        return response_list