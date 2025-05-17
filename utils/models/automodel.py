import json
import os


class AutoModel:
    @classmethod
    def from_pretrained(cls, model_path=None, **kwargs):
        assert model_path is not None, "Please provide model path"
        if os.path.isdir(model_path):
            if os.path.exists(os.path.join(model_path, "config.json")):
                with open(os.path.join(model_path, "config.json"), "r") as f:
                    config = json.load(f)

                    architectures = config.get("architectures", None)
                    if architectures is None:
                        language_config = config.get("language_config", None)
                        if language_config is not None:
                            architectures = language_config.get("architectures", None)
                    if architectures is not None:
                        if "Qwen2VLForConditionalGeneration" in architectures:
                            from .qwen2vl import Qwen2VL
                            return Qwen2VL(model_path=model_path, **kwargs)
                        elif "InternVLChatModel" in architectures:
                            from .internvl2 import InternVL2
                            import nest_asyncio
                            nest_asyncio.apply()
                            return InternVL2(model_path=model_path, **kwargs)
                        elif "MiniCPMV" in architectures:
                            from .minicpmv import MiniCPMV
                            return MiniCPMV(model_path=model_path, **kwargs)
                        elif "DeepseekV2ForCausalLM" in architectures:
                            from .deepseekvl2 import DeepSeekVL2
                            return DeepSeekVL2(model_path=model_path, **kwargs)
                        elif "LlavaQwenForCausalLM" in architectures:
                            from .llava_onevision import LLaVAOneVision
                            return LLaVAOneVision(model_path=model_path, **kwargs)
                        else:
                            raise NotImplementedError("Unsupported model architecture: {}".format(architectures))