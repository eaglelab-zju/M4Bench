import gc
import torch

from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

class Qwen2VL:
    def __init__(self, model_path=None, min_pixels=256*28*28, max_pixels=1280*28*28, do_sample=False, max_new_tokens=1024, top_k=10, top_p=0.001, temperature=0.1, repetition_penalty=1.0):
        assert model_path is not None, "Please provide model path"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype="auto", 
            device_map="auto"
        )

        self.processor = AutoProcessor.from_pretrained(
            model_path, 
            min_pixels=min_pixels, 
            max_pixels=max_pixels
        )
        
        if not do_sample:
            self.gen_config = {
                "do_sample": False,
                "max_new_tokens": max_new_tokens,
                "top_k": None,
                "top_p": None,
                "temperature": None,
                "repetition_penalty": repetition_penalty
            }
        else:
            self.gen_config = {
                "do_sample": True,
                "max_new_tokens": max_new_tokens,
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty
            }
    
    def parse_input(self, query=None, imgs=None):
        if imgs is None:
            messages = [
                {"role": "user", "content": query},
            ]
            return messages
        
        if isinstance(imgs, str):
            imgs = [imgs]
        
        content = []
        for img in imgs:
            content.append({
                "type": "image",
                "image": img
            })
        content.append({
            "type": "text",
            "text": query
        })

        messages = [
            {"role": "user", "content": content},
        ]
        return messages
    
    def generate(self, messages=None, batch=False):
        if not batch:
            texts = [
                self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=True)
            ]
        else:
            texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, add_vision_id=True)
                for msg in messages
            ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(**inputs, **self.gen_config)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response_list = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()
        gc.collect()

        return response_list

    def infer(self, query=None, imgs=None):
        messages = self.parse_input(query, imgs)
        response = self.generate(messages=messages, batch=False)[0]
        return response
                
    def batch_infer(self, query_list=None, imgs_list=None):
        messages_list = [self.parse_input(query, imgs) for query, imgs in zip(query_list, imgs_list)]
        response_list = self.generate(messages=messages_list, batch=True)
        return response_list
        
    def chat(self, query=None, imgs=None, history=None):
        if history is None:
            history = []
        
        new_messages = self.parse_input(query, imgs)
        history.extend(new_messages)
        response = self.generate(history, batch=False)[0]
        history.append({
            "role": "assistant",
            "content": response
        })
        return response, history