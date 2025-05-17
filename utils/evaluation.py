import copy as cp
import os
import random as rd
import string

from dotenv import load_dotenv
from openai import OpenAI

def cn_string(s):
    import re
    if re.search(u'[\u4e00-\u9fff]', s):
        return True
    return False

def build_prompt_blink(question, options, prediction):
    tmpl = (
        'You are an AI assistant who will help me to match an answer with several options of a single-choice question. '
        'You are provided with a question, several options, and an answer, '
        'and you need to find which option is most similar to the answer. '
        "If the answer says things like refuse to answer, I'm sorry cannot help, etc., output Z."
        'If the meaning of all options are significantly different from the answer, '
        'or the answer does not select any option, output Z. '
        'Your should output one of the choices, A, B, C, D (if they are valid options), or Z.\n'
        'Example 1: \n'
        'Question: Which point is closer to the camera?\nSelect from the following choices.\n'
        'Options: A. Point A\nB. Point B\n(Z) Failed\n'
        'Answer: Point B, where the child is sitting, is closer to the camera.\nYour output: (B)\n'
        'Example 2: \n'
        'Question: Which point is closer to the camera?\nSelect from the following choices.\n'
        'Options: (A) Point A\n(B) Point B\n(Z) Failed\n'
        "Answer: I'm sorry, but I can't assist with that request.\nYour output: (Z)\n"
        'Example 3: \n'
        'Question: Which point is corresponding to the reference point?\nSelect from the following choices.\n'
        'Options: (A) Point A\n(B) Point B\n(Z) Failed\n'
        'Answer:The reference point (REF) on the first image is at the tip of the pot, '
        'which is the part used to Poke if the pots were used for that action. Looking at the second image, '
        'we need to find the part of the object that would correspond to poking.\n'
        "(A) Point A is at the tip of the spoon's handle, which is not used for poking.\n"
        '(B) Point B is at the bottom of the spoon, which is not used for poking.\n'
        '(C) Point C is on the side of the pspoonot, which is not used for poking.\n'
        '(D) Point D is at the tip of the spoon, which is not used for poking.\n'
        '\nTherefore, there is no correct answer in the choices\nYour output: (Z)\n'
        'Example 4: \n'
        'Question: {}\nOptions: {}\n(Z) Failed\nAnswer: {}\nYour output: '
    )
    return tmpl.format(question, options, prediction)

def build_prompt_cn(question, options, prediction):
    tmpl = (
        '你是一个帮助我匹配答案与单选题中多个选项的 AI 助手。'
        '你会被提供：一个问题，多个选项，一个答案。你的任务是找到与答案意义最相近的选项。'
        '如果所有选项的意义都与答案显著不同，则输出 Z。'
        '你应该输出一个单个的大写字母，例如 A, B, C, D（如果它们是有效选项），或 Z。'
        '例 1:'
        '问题: 图中最主要的物体是什么?\n选项: A. 泰迪熊 B. 兔子 C. 猫 D. 狗\n答案: 一只可爱的泰迪熊\n输出: A\n'
        '例 2: \n'
        '问题: 图中最主要的物体是什么?\n选项: A. 泰迪熊 B. 兔子 C. 猫 D. 狗\n答案: 蜘蛛\n输出: Z\n'
        '例 3: \n'
        '问题: {}\n选项: {}\n答案: {}\n输出: '
    )
    return tmpl.format(question, options, prediction)

def build_prompt(question, options, prediction):
    tmpl = (
        'You are an AI assistant who will help me to match '
        'an answer with several options of a single-choice question. '
        'You are provided with a question, several options, and an answer, '
        'and you need to find which option is most similar to the answer. '
        'If the meaning of all options are significantly different from the answer, output Z. '
        'Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n'
        'Example 1: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: a cute teddy bear\nYour output: A\n'
        'Example 2: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: Spider\nYour output: Z\n'
        'Example 3: \n'
        'Question: {}\nOptions: {}\nAnswer: {}\nYour output: '
    )
    return tmpl.format(question, options, prediction)

def build_choices(item):
    ret = {}
    for ch in string.ascii_uppercase:
        if ch in item:
            ret[ch] = item[ch]
    return ret

def build_option_str(option_dict):
    s = 'There are several options: \n'
    for c, content in option_dict.items():
        s += f'{c}. {content}\n'
    return s

def can_infer_text(answer, choices):
    answer = answer.lower()
    assert isinstance(choices, dict)
    for k in choices:
        assert k in string.ascii_uppercase
        choices[k] = str(choices[k]).lower()
    cands = []
    for k in choices:
        if choices[k] in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return False

def can_infer_option(answer, choices):
    verbose = os.environ.get('VERBOSE', 0)
    if 'Failed to obtain answer via API' in answer:
        return False

    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        'Cannot determine the answer'
    ]
    for err in reject_to_answer:
        if err in answer:
            return 'Z'

    def count_choice(splits, choices, prefix='', suffix=''):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = cp.copy(answer)
    chars = '.()[],:;!*#{}'
    for c in chars:
        answer_mod = answer_mod.replace(c, ' ')

    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            if 'A' in splits and len(splits) > 3 and verbose:
                print(f'A might be a quantifier in the string: {answer}.')
                return False
            if ch in splits:
                return ch
    elif count == 0 and count_choice(splits, {'Z', ''}) == 1:
        return 'Z'
    return False

def can_infer(answer, choices):
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    return copt if copt else can_infer_text(answer, choices)

def extract_answer_from_item(model, item, dataset_name=None):
    choices = build_choices(item)
    option_str = build_option_str(choices)

    if dataset_name == 'BLINK':
        prompt = build_prompt_blink(item['question'], option_str, item['prediction'])
    elif cn_string(item['question']):
        prompt = build_prompt_cn(item['question'], option_str, item['prediction'])
    else:
        prompt = build_prompt(item['question'], option_str, item['prediction'])

    ret = can_infer(item['prediction'], choices)
    if ret:
        return dict(opt=ret, log=item['prediction'])

    if model is None:
        return dict(opt='Z', log='Failed in Prefetch, no GPT-based answer matching under `exact_matching` policy.')

    retry = 3
    while retry:
        ans = model.infer(prompt)
        if 'Failed to obtain answer via API' in ans:
            print('GPT API failed to answer. ')
        else:
            ret = can_infer(ans, choices)
            if ret:
                return dict(opt=ret, log=ans)
            else:
                print(f'Output includes 0 / > 1 letter among candidates {set(choices)} and Z: {ans}')
        retry -= 1

        if retry == 0:
            options = list(choices) + ['Z'] if 'Z' not in choices else []
            return dict(opt=rd.choice(options), log='Failed to predict, thus randomly generate one.')

class GPT:
    def __init__(self, model_name = "gpt-3.5-turbo", base_url = "https://api.openai.com/v1"):
        api_key = os.environ.get("OPENAI_API_KEY", None)
        if api_key is None:
            raise ValueError("Please set OPENAI_API_KEY environment variable.")
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url

    def infer(self, prompt):
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

