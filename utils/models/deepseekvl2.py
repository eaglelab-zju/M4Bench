import gc
import torch

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images
from transformers import AutoModelForCausalLM

class DeepSeekVL2:
    def __init__(self, model_path=None, do_sample=False, max_new_tokens=1024, top_k=20, top_p=0.8, temperature=1.0, repetition_penalty=1.0):
        assert model_path is not None, "Please provide model path"
        
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype="auto", device_map="auto")
        self.processor = DeepseekVLV2Processor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer

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
            imgs = []

        if isinstance(imgs, str):
            imgs = [imgs]
        
        conversation = []
        user_content = ""
        for index, image in enumerate(imgs):
            user_content += f"This is image_{index+1}: <image>\n"
        user_content += f"{query}\n"
        conversation.append({
            "role": "<|User|>",
            "content": user_content,
            "images": imgs
        })

        conversation.append({
            "role": "<|Assistant|>",
            "content": ""
        })
        return conversation
    
    def infer(self, query=None, imgs=None):
        self.model.eval()
        conversation = self.parse_input(query, imgs)
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(self.model.device)

        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        outputs = self.model.language.generate(
            # input_ids=prepare_inputs['input_ids'],
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **self.gen_config,
            use_cache=True,
        )
        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        del prepare_inputs, inputs_embeds, outputs
        torch.cuda.empty_cache()
        gc.collect()
        return answer