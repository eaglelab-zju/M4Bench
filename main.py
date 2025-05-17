import argparse
import copy
import json
import os
import random
import re
import warnings

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from termcolor import colored
from tqdm import tqdm
from utils.evaluation import GPT, extract_answer_from_item
from utils.common_utils import seed_everything
from utils.prompt import prompt_process_mapping
from utils.models import AutoModel

# model_name: (model_path, need_remove_image_tag, grounding_template)
model_mapping = {
    "Qwen2-VL-2B-Instruct": ("llm/Qwen/Qwen2-VL-2B-Instruct", True, 'qwen2vl'),
    "Qwen2-VL-7B-Instruct": ("llm/qwen/Qwen2-VL-7B-Instruct", True, 'qwen2vl'),
    "InternVL2-4B": ("llm/OpenGVLab/InternVL2-4B", True, "internvl2"),
    "InternVL2-8B": ("llm/OpenGVLab/InternVL2-8B", True, "internvl2"),
    "InternVL2_5-4B": ("llm/OpenGVLab/InternVL2_5-4B", True, "internvl2"),
    "InternVL2_5-8B": ("llm/OpenGVLab/InternVL2_5-8B", True, "internvl2"),
    "deepseek-vl2-tiny": ("llm/deepseek-ai/deepseek-vl2-tiny", True, 'deepseek-vl2'),
    "deepseek-vl2-small": ("llm/deepseek-ai/deepseek-vl2-small", True, 'deepseek-vl2'),
    "MiniCPM-V-2_6": ("llm/OpenBMB/MiniCPM-V-2_6", True, None),
    "llava-onevision-qwen2-7b-ov-chat": ("llm/lmms-lab/llava-onevision-qwen2-7b-ov-chat", True, None),
    "GPT-4o": ("API Key", True, "closed_source"),
    "Gemini-Pro": ("API Key", True, "closed_source"),
    "Qwen-VL-Max": ("API Key", True, "closed_source")
}
# task_name: (task_path, task_type)
task_mapping = {
    "Object_States": ("dataset/Object_States/instruction.json", "MCQ"),
    "State_Invariance": ("dataset/State_Invariance/instruction.json", "MCQ"),
    "Detailed_Difference_Generated_Images" : ("dataset/Detailed_Difference/instruction_generated_images.json", "MCQ"),
    "Detailed_Difference_Natural_Images": ("dataset/Detailed_Difference/instruction_nature_images.json", "MCQ"),
    "Spatial_Perception_With_Visual_Prompt": ("dataset/Spatial_Perception/instruction.json", "MCQ"),
    "Spatial_Perception_WithOut_Visual_Prompt": ("dataset/Spatial_Perception/instruction.json", "MCQ"),
    "Instance_Comparison_With_Visual_Prompt": ("dataset/Instance_Comparison/instruction.json", "MCQ"),
    "Instance_Comparison_WithOut_Visual_Prompt": ("dataset/Instance_Comparison/instruction.json", "MCQ"),
}

def eval_task(task_name=None, model_name=None, remove_image_tag=True, grounding_template=None, save_path=None, model=None, gpt=None):
    # step-0, initialize the task results
    sample_list = []
    results = {
        "model_name": model_name,
        "task_name": task_name,
        "score": None
    }
    if isinstance(model, str):
        pre_test_result_dir = 'MFourBench/close_mllms_csv_output'
        if task_name == 'Detailed_Difference_Generated_Images':
            task_name_lower = 'detailed_difference_gi'
        elif task_name == 'Detailed_Difference_Natural_Images':
            task_name_lower = 'detailed_difference_ni'
        elif task_name == 'Object_States':
            task_name_lower = 'object_states'
        elif task_name == 'State_Invariance':
            task_name_lower = 'si_close_mllms'
        elif task_name == 'Spatial_Perception_With_Visual_Prompt':
            task_name_lower = 'sp_with_visual_prompts'
        elif task_name == 'Spatial_Perception_WithOut_Visual_Prompt':
            task_name_lower = 'sp_without_visual_prompts'
        elif task_name == 'Instance_Comparison_With_Visual_Prompt':
            task_name_lower = 'ic_with_visual_prompts'
        elif task_name == 'Instance_Comparison_WithOut_Visual_Prompt':
            task_name_lower = 'ic_without_visual_prompts'
        else:
            pass

        if model_name == 'GPT-4o':
            model_name_lower = 'gpt4'
        elif model_name == 'Gemini-Pro':
            model_name_lower = 'gemini'
        elif model_name == 'Qwen-VL-Max':
            model_name_lower = 'qwenmax'
        else:
            pass
        pre_test_result_path = os.path.join(pre_test_result_dir, task_name_lower, f"{task_name_lower}_{model_name_lower}_result.json")
        assert os.path.exists(pre_test_result_path), f"{pre_test_result_path} does not exist"
        with open(pre_test_result_path, "r") as f:
            pre_test_result_data = json.load(f)
    
    # step-1, prepare the dataset
    task_path, task_type = task_mapping[task_name]
    with open(task_path, "r") as f:
        dataset = json.load(f)
    
    # step-2, prepare the metric
    if task_type == "MCQ":
        acc_cnt = 0
    else:
        pass

    for i, sample in enumerate(tqdm(dataset, desc=task_name, unit="sample")):
        prompt, question, options_dict = prompt_process_mapping[task_name](sample, remove_image_tag, grounding_template)
        # images = sample['image']
        if 'image' not in sample:
            images = sample['images']
        else:
            images = sample['image']
        answer = sample['answer']
        if isinstance(model, str):
            pred = pre_test_result_data[i]['model_full_prediction']
        else:
            pred = model.infer(prompt, images)
        pred_item = copy.deepcopy(options_dict)
        pred_item['question'] = question
        pred_item['prediction'] = pred

        if task_type == "MCQ":
            model_extracted_dict = extract_answer_from_item(
                gpt, pred_item, task_name
            )
            log = model_extracted_dict['log']
            if log != 'Failed to predict, thus randomly generate one.':
                model_extracted = model_extracted_dict['opt']
            else:
                model_extracted = 'Z'
            if model_extracted == answer:
                acc_cnt += 1
            sample_list.append({
                "id": sample['id'],
                "prompt": prompt,
                "images": images,
                "answer": answer,
                "model_full_prediction": pred,
                "model_extracted_answer": model_extracted
            })
            tqdm.write(
                (
                    f"id: {sample['id']}\n"
                    f"model prediction: {pred}\n"
                    f"model extracted answer: {model_extracted}\n"
                    f"answer: {answer}\n"
                )
            )
    
    results['results'] = sample_list
    if task_type == "MCQ":
        results["score"] = acc_cnt / len(dataset) if dataset else 0
    else:
        pass

    with open(save_path, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(colored(f"{model_name} on {task_name} saved to: {save_path}", "red"))
    print(colored(f"{model_name} on {task_name} score: {results['score']}", "blue"))

if __name__ == "__main__":
    # step-0, initialize the evaluation environment
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task_list", type=str, default="all")
    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    load_dotenv()

    # step-1, prepare the dataset
    if args.task_list == "all":
        task_list = task_mapping.keys()
    else:
        task_list = [task.strip() for task in args.task_list.split(",")]
        invalid_task_list = [task for task in task_list if task not in task_mapping.keys()]
        if len(invalid_task_list) > 0:
            raise ValueError(f"Invalid task name: {', '.join(invalid_task_list)}; MFourBench only supports {', '.join(task_mapping.keys())}")

    # step-2, prepare the model and gpt
    if args.model_name not in model_mapping.keys():
        raise ValueError(f"Invalid model name: {args.model_name}; MFourBench only supports {', '.join(model_mapping.keys())}")
    if os.environ.get("OPENAI_API_KEY", None):
        gpt = GPT(model_name='gpt-3.5-turbo')
    else:
        gpt = None
    model_path, remove_image_tag, grounding_template = model_mapping[args.model_name]
    if model_path != "API Key":
        model = AutoModel.from_pretrained(model_path, do_sample=False, repetition_penalty=1.0)
    else:
        model = "API Key"

    # step-3, evaluate the task
    for task in task_list:
        save_path = os.path.join(args.save_dir, f"{args.model_name}/{task}.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        eval_task(
            task_name=task,
            model_name=args.model_name,
            remove_image_tag=remove_image_tag,
            grounding_template=grounding_template,
            save_path=save_path,
            model=model,
            gpt=gpt
        )