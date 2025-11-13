import random
import numpy as np
import torch
from time import sleep
import torch
import math

import torchvision.transforms as T

from PIL import Image
from transformers import AutoModel, AutoTokenizer
from torchvision.transforms.functional import InterpolationMode
import dataclasses
from enum import IntEnum, auto
from typing import Dict, List, Tuple, Union
import json
import re


def set_cudnn(device="cuda"):
    torch.backends.cudnn.enabled = device == "cuda"
    torch.backends.cudnn.benchmark = device == "cuda"


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def stop_epoch(time=3):
    try:
        print("can break now")
        for i in range(time):
            sleep(1)
        print("wait for next epoch")
        return False
    except KeyboardInterrupt:
        return True


def compute_loss_accuracy(net, data_loader, criterion, device):
    net.eval()
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            total_loss += criterion(outputs, labels).item()
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()

    return total_loss / (batch_idx + 1), correct / len(data_loader.dataset)


def create_dataset_mapping(json_file_path):
    """
    从JSON文件中提取所有唯一的dataset名称，并创建一个从0开始递增的数字映射字典

    参数:
    json_file_path: JSON文件路径

    返回:
    一个字典，格式为 {dataset_name1: 0, dataset_name2: 1, ...}
    """
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取所有唯一的dataset名称
    unique_datasets = set()
    for item in data:
        if "dataset" in item:
            unique_datasets.add(item["dataset"])
        elif "id" in item:
            unique_datasets.add(str(item["id"]))

    # 创建映射字典（按字母排序）
    sorted_datasets = sorted(list(unique_datasets))
    mapping = {dataset: idx for idx, dataset in enumerate(sorted_datasets)}

    return mapping


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=6):
    image = Image.open(image).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def split_response(response, sep='\n\n', max_steps=None):
    steps = response.split(sep)

    if max_steps is not None:
        step = math.ceil(len(steps) / max_steps)
        new_steps = []
        for i in range(0, len(steps), step):
            new_steps.append(sep.join(steps[i:i + step]))
        return new_steps

    return steps

def find_placeholder_idx(template, tokenizer, input_ids, PLACEHOLDER):
    # TODO: support batch inference
    input_ids = input_ids[0].tolist()
    template = get_conv_template(template)

    idx = []
    bos =  tokenizer(template.roles[1], add_special_tokens=False).input_ids
    target = tokenizer(template.roles[1] + PLACEHOLDER + template.sep, add_special_tokens=False).input_ids
    for i in range(len(input_ids)):
        if input_ids[i:i+len(target)] == target:
            assert i + len(bos) - 1 >= 0
            idx.append(i + len(bos) - 1)

    return idx

class SeparatorStyle(IntEnum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    NO_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()
    LLAMA2 = auto()
    CHATGLM = auto()
    CHATML = auto()
    CHATINTERN = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    ROBIN = auto()
    FALCON_CHAT = auto()
    CHATGLM3 = auto()
    INTERNVL_ZH = auto()
    MPT = auto()


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = '{system_message}'
    # The system message
    system_message: str = ''
    # The names of two roles
    roles: Tuple[str] = ('USER', 'ASSISTANT')
    # All messages. Each item is (role, message).
    messages: List[List[str]] = ()
    # The number of few shot examples
    offset: int = 0
    # The separator style and configurations
    sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE
    sep: str = '\n'
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ': ' + message + self.sep
                else:
                    ret += role + ':'
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ': ' + message + seps[i % 2]
                else:
                    ret += role + ':'
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ': ' + message + self.sep
                else:
                    ret += role + ': '  # must be end with a space
            return ret
        elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            ret = '' if system_prompt == '' else system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + '\n' + message + self.sep
                else:
                    ret += role + '\n'
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.RWKV:
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += (
                        role
                        + ': '
                        + message.replace('\r\n', '\n').replace('\n\n', '\n')
                    )
                    ret += '\n\n'
                else:
                    ret += role + ':'
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA2:
            seps = [self.sep, self.sep2]
            if self.system_message:
                ret = system_prompt
            else:
                ret = '[INST] '
            for i, (role, message) in enumerate(self.messages):
                tag = self.roles[i % 2]
                if message:
                    if i == 0:
                        ret += message + ' '
                    else:
                        ret += tag + ' ' + message + seps[i % 2]
                else:
                    ret += tag
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM:
            # source: https://huggingface.co/THUDM/chatglm-6b/blob/1d240ba371910e9282298d4592532d7f0f3e9f3e/modeling_chatglm.py#L1302-L1308
            # source2: https://huggingface.co/THUDM/chatglm2-6b/blob/e186c891cf64310ac66ef10a87e6635fa6c2a579/modeling_chatglm.py#L926
            round_add_n = 1 if self.name == 'chatglm2' else 0
            if system_prompt:
                ret = system_prompt + self.sep
            else:
                ret = ''

            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += f'[Round {i//2 + round_add_n}]{self.sep}'

                if message:
                    ret += f'{role}：{message}{self.sep}'
                else:
                    ret += f'{role}：'
            return ret
        elif self.sep_style == SeparatorStyle.CHATML:
            ret = '' if system_prompt == '' else system_prompt + self.sep + '\n'
            for role, message in self.messages:
                if message:
                    ret += role + '\n' + message + self.sep + '\n'
                else:
                    ret += role + '\n'
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM3:
            ret = ''
            if self.system_message:
                ret += system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + '\n' + ' ' + message
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.CHATINTERN:
            # source: https://huggingface.co/internlm/internlm-chat-7b-8k/blob/bd546fa984b4b0b86958f56bf37f94aa75ab8831/modeling_internlm.py#L771
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                # if i % 2 == 0:
                #     ret += "<s>"
                if message:
                    ret += role + ':' + message + seps[i % 2] + '\n'
                else:
                    ret += role + ':'
            return ret
        elif self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ':\n' + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += '\n\n'
                else:
                    ret += role + ':\n'
            return ret
        elif self.sep_style == SeparatorStyle.PHOENIX:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + ': ' + '<s>' + message + '</s>'
                else:
                    ret += role + ': ' + '<s>'
            return ret
        elif self.sep_style == SeparatorStyle.ROBIN:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ':\n' + message + self.sep
                else:
                    ret += role + ':\n'
            return ret
        elif self.sep_style == SeparatorStyle.FALCON_CHAT:
            ret = ''
            if self.system_message:
                ret += system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ': ' + message + self.sep
                else:
                    ret += role + ':'

            return ret
        elif self.sep_style == SeparatorStyle.INTERNVL_ZH:
            seps = [self.sep, self.sep2]
            ret = self.system_message + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ': ' + message + seps[i % 2]
                else:
                    ret += role + ':'
            return ret
        elif self.sep_style == SeparatorStyle.MPT:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f'Invalid style: {self.sep_style}')

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.
        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{'role': 'system', 'content': self.system_message}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({'role': 'user', 'content': msg})
            else:
                if msg is not None:
                    ret.append({'role': 'assistant', 'content': msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            'template_name': self.name,
            'system_message': self.system_message,
            'roles': self.roles,
            'messages': self.messages,
            'offset': self.offset,
        }


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert (
            template.name not in conv_templates
        ), f'{template.name} has been registered.'

    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()


# Both Hermes-2 and internlm2-chat are chatml-format conversation templates. The difference
# is that during training, the preprocessing function for the Hermes-2 template doesn't add
# <s> at the beginning of the tokenized sequence, while the internlm2-chat template does.
# Therefore, they are completely equivalent during inference.
register_conv_template(
    Conversation(
        name='Hermes-2',
        system_template='<|im_start|>system\n{system_message}',
        # note: The new system prompt was not used here to avoid changes in benchmark performance.
        # system_message='我是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。',
        system_message='你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。',
        roles=('<|im_start|>user\n', '<|im_start|>assistant\n'),
        sep_style=SeparatorStyle.MPT,
        sep='<|im_end|>',
        stop_str='<|endoftext|>',
    )
)


register_conv_template(
    Conversation(
        name='internlm2-chat',
        system_template='<|im_start|>system\n{system_message}',
        # note: The new system prompt was not used here to avoid changes in benchmark performance.
        # system_message='我是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。',
        system_message='你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。',
        roles=('<|im_start|>user\n', '<|im_start|>assistant\n'),
        sep_style=SeparatorStyle.MPT,
        sep='<|im_end|>',
    )
)


register_conv_template(
    Conversation(
        name='phi3-chat',
        system_template='<|system|>\n{system_message}',
        # note: The new system prompt was not used here to avoid changes in benchmark performance.
        # system_message='我是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。',
        system_message='你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。',
        roles=('<|user|>\n', '<|assistant|>\n'),
        sep_style=SeparatorStyle.MPT,
        sep='<|end|>',
    )
)

register_conv_template(
    Conversation(
        name='internvl2_5',
        system_template='<|im_start|>system\n{system_message}',
        system_message='你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。',
        roles=('<|im_start|>user\n', '<|im_start|>assistant\n'),
        sep_style=SeparatorStyle.MPT,
        sep='<|im_end|>\n',
    )
)

def input_processing(
        self,
        tokenizer,
        prompt,
        pixel_values,
        num_patches_list=None,
        max_steps=None,
        IMG_START_TOKEN='<img>',
        IMG_END_TOKEN='</img>',
        IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
):
    if pixel_values is not None and '<image>' not in prompt:
        num_images = 1 if num_patches_list is None else len(num_patches_list)
        # 模式（要查找的子串）
        pattern = "<|im_start|>user\n"

        # 找到第一个 pattern 的位置
        idx = prompt.find(pattern)
        if idx == -1:
            # 如果没找到，直接在末尾追加
            prompt = prompt + '<image>'
        else:
            # 计算插入点
            insert_pos = idx + len(pattern)
            # 构造新字符串
            prompt = prompt[:insert_pos] + '<image>' + prompt[insert_pos:]

    if num_patches_list is None:
        num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []

    assert pixel_values is None or (
                len(pixel_values) == sum(num_patches_list) and len(num_patches_list) == prompt.count(
            '<image>')), f'{len(pixel_values)=}, {sum(num_patches_list)=}, {len(num_patches_list)}, {prompt=}'

    image_input = pixel_values is not None
    if pixel_values is None:
        pixel_values = torch.zeros(1, 3, self.config.vision_config.image_size, self.config.vision_config.image_size).to(
            self.device).to(torch.bfloat16)

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    self.img_context_token_id = img_context_token_id

    query = prompt

    for num_patches in num_patches_list:
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
        query = query.replace('<image>', image_tokens, 1)

    # Prepare inputs
    model_inputs = tokenizer(query, return_tensors='pt')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = self.device
    input_ids = model_inputs['input_ids'].to(device)
    attention_mask = model_inputs['attention_mask'].to(device)
    image_flags = torch.tensor([image_input] * pixel_values.size(0), dtype=torch.long).to(device)
    pixel_values = pixel_values.to(torch.bfloat16).to(device)

    return input_ids, attention_mask, image_flags, pixel_values

def generate_target(
        input_ids, # (batch, seq_len)
        tokenizer,
        template,
        PLACEHOLDER=None,
        str2score=None,
):
    # TODO: support batch inference
    if str2score is None:
        str2score = {'+': 1, '-': 0}

    if PLACEHOLDER is None:
        PLACEHOLDER = '+'
        PLACEHOLDER_2 = '-'

    candidate_tokens = []
    candidate_weights = []

    # Prepare Query
    for k, v in str2score.items():
        k_id = tokenizer.convert_tokens_to_ids(k)
        assert k_id != tokenizer.unk_token_id

        candidate_tokens.append(k_id)
        candidate_weights.append(v)

    idx = find_placeholder_idx(template, tokenizer, input_ids, PLACEHOLDER=PLACEHOLDER)
    idx_2 = find_placeholder_idx(template, tokenizer, input_ids, PLACEHOLDER=PLACEHOLDER_2)
    # 初始化全为 -1（表示不监督）
    target = torch.full_like(input_ids, -1)

    # 对所有 batch 设置目标值（如果你只想监督第 0 个 sample，可以改成 target[0, idx]）
    target[:, idx] = candidate_tokens[0]
    target[:, idx_2] = candidate_tokens[1]
    return target


def aggregate_score(logits: torch.Tensor, target: torch.Tensor, func):
    """
    打印每个监督位置的目标 token 的 logits 值。

    参数：
        logits: Tensor，形状 [batch_size, seq_len, vocab_size]
        target: Tensor，形状 [batch_size, seq_len]，未监督位置填 -1
    """
    # 计算 mask：哪些位置需要监督
    mask = (target != -1)  # [B, T]

    # 获取监督位置的 batch 和 seq 索引
    batch_indices, seq_indices = torch.nonzero(mask, as_tuple=True)

    # 获取这些位置对应的目标 token id
    token_indices = target[batch_indices, seq_indices]

    # 获取对应位置的 logits 值
    supervised_logits = logits[batch_indices, seq_indices, token_indices]

    if func == 'mean':
        return torch.mean(supervised_logits, dim=0)


def aggregate_score_negative(logits: torch.Tensor, target: torch.Tensor, func, label):
    """
    打印每个监督位置的目标 token 的 logits 值。

    参数：
        logits: Tensor，形状 [batch_size, seq_len, vocab_size]
        target: Tensor，形状 [batch_size, seq_len]，未监督位置填 -1
    """
    # 计算 mask：哪些位置需要监督
    mask = (target != -1)  # [B, T]

    # 负向量
    if label == 0:
        target[target == 10] = 12

    # 获取监督位置的 batch 和 seq 索引
    batch_indices, seq_indices = torch.nonzero(mask, as_tuple=True)

    # 获取这些位置对应的目标 token id
    token_indices = target[batch_indices, seq_indices]

    # 获取对应位置的 logits 值
    supervised_logits = logits[batch_indices, seq_indices, token_indices]

    if func == 'mean':
        return torch.mean(supervised_logits, dim=0)


SYSTEM_PROMPT = (
    "You are an advanced AI assistant, designed to serve as a process supervision model. "
    "In this task, I will provide a problem statement followed by the first step of the solution process. "
    "For each subsequent turn, I will give you a new step in the solution. Your role is to assess "
    "whether the solution process is correct up to the current step.\n\n"
    "- In the **first round**, I will input the problem and the first step of the solution process.\n"
    "- In **each subsequent round**, I will provide the next step in the solution.\n\n"
    "For each step, you should:\n"
    "- Respond with **\"+\"** if you believe the solution process is correct up to this step.\n"
    "- Respond with **\"-\"** if you detect any issues or errors in the process up to this step.\n\n"
    "Please note:\n"
    "- Only respond with **\"+\"** or **\"-\"**. Do not provide any additional explanations, comments, or justifications.\n\n"
    "Your task is to verify the accuracy and correctness of each step in the given solution process."
)

# 行首顶层步骤（与原意一致，但更宽松）
TOP_STEP_PAT = re.compile(
    r"^\s*(?:"
    r"(?:Step|步骤)\s*\d+\s*[:：\.]?|"     # Step 1: / 步骤1：
    r"\d+\s*[\.、\)]|"                    # 1. / 1、 / 1)
    r"[①-⑳]"                              # ① 到 ⑳
    r")\s*",
    re.IGNORECASE
)

# 行内的枚举分隔标记（用于把“1) foo 2) bar 3) baz”一行切开）
INLINE_STEP_PAT = re.compile(
    r"(?:^|\s)(?:(?:Step|步骤)\s*\d+\s*[:：\.]?|\d+\s*[\.、\)]|[①-⑳])\s+",
    re.IGNORECASE
)

BLANK_BLOCK_SPLIT = re.compile(r"(?:\r?\n\s*){2,}")  # 按≥1个空行分块

def _strip_leading_marker(s: str) -> str:
    return TOP_STEP_PAT.sub("", s, count=1).strip()

def _split_inline_enums(text: str) -> List[str]:
    """把一段里串联的 1)/2)/3) 拆成多段；若检测不到则原样返回单段"""
    parts = []
    it = list(INLINE_STEP_PAT.finditer(text))
    if not it:
        return [text.strip()] if text.strip() else []

    # 如果开头没有匹配，从开头补一个“虚拟起点”
    starts = []
    if it[0].start() != 0:
        starts.append(0)
    starts += [m.start() for m in it]

    # 计算每段的起止
    for i, st in enumerate(starts):
        en = starts[i+1] if i+1 < len(starts) else len(text)
        seg = text[st:en].strip()
        if not seg:
            continue
        # 去掉段首可能带的标记
        seg = _strip_leading_marker(seg)
        if seg:
            parts.append(seg)
    return parts or ([text.strip()] if text.strip() else [])

def split_reasoning_block(block: str) -> List[str]:
    block = (block or "").strip()
    if not block:
        return [""]

    lines = block.splitlines()

    steps, cur = [], []

    def flush():
        nonlocal cur
        if cur:
            seg = "\n".join(cur).rstrip()
            if seg.strip():
                steps.append(seg)
            cur = []

    # 第一阶段：优先用“行首顶层编号”切分
    has_top_mark = sum(bool(TOP_STEP_PAT.match(ln)) for ln in lines) >= 2
    if has_top_mark:
        for i, ln in enumerate(lines):
            if TOP_STEP_PAT.match(ln) and (i == 0 or not lines[i-1].strip()):
                flush()
                cur.append(_strip_leading_marker(ln))
            else:
                cur.append(ln.rstrip())
        flush()
    else:
        # 第二阶段：按空行分块
        blocks = [b for b in BLANK_BLOCK_SPLIT.split(block) if b.strip()]
        if len(blocks) >= 2:
            steps = blocks
        else:
            # 实在没有空行分块，就退化为整块
            steps = [block]

    # 第三阶段：对每个步骤做“行内枚举”再细分
    final_steps: List[str] = []
    for seg in steps:
        # 如果这个段落本身没有明显行首编号，但包含多个行内编号，则拆开
        inline_parts = _split_inline_enums(seg)
        # 再把每个 part 的行首标记去掉（保险）
        for p in inline_parts:
            cleaned = _strip_leading_marker(p)
            if cleaned:
                final_steps.append(cleaned)

    return final_steps or [""]

# ---------- Conversation 构造 ----------
def build_conversations(question: str, steps: List[str]) -> List[Dict]:
    conv = [{"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human",  "value": "### Question:\n" + question.strip()}]
    for st in steps:
        conv.append({"from": "human", "value": st})
        conv.append({"from": "gpt",   "value": "+"})
    return conv


def select_best_answer(model, tokenizer, inputs, agg_fuc):
    # Todo: support batch inference
    pixel_values = load_image(inputs['image_path'][0], max_num=12).to(torch.bfloat16).cuda()
    question = inputs['question'][0]
    index = 0
    max_score = 0
    true_false = False
    best_index = 0
    for i in inputs['candidates']:
        m_reason = re.search(r"\[Reasoning\](.*?)(?=\n?Answer:)", i[0], re.S)
        reasoning = m_reason.group(1) if m_reason else i[0]
        m_ans = re.search(r"Answer:\s*(.*)", i[0], re.S)
        steps = split_reasoning_block(reasoning)
        answer = m_ans.group(1).strip() if m_ans else ""
        if answer:
            steps.append(f"Answer: {answer}")
        conversation = build_conversations(question, steps)
        conv_template = get_conv_template(model.template)
        for part in conversation:
            if part['from'] == 'system':
                conv_template.system_message = part['value']
            elif part['from'] == 'human':
                conv_template.append_message(conv_template.roles[0], part['value'])
            elif part['from'] == 'gpt':
                conv_template.append_message(conv_template.roles[1], part['value'])
        prompt = conv_template.get_prompt()
        input_ids, attention_mask, image_flags, pixel_values = input_processing(model, tokenizer, prompt, pixel_values)
        output = model(pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,).logits  # (batch, seq_len, vocab_size)
        probs = torch.softmax(output, dim=-1)
        target = generate_target(input_ids, tokenizer, model.template)  # (batch, seq_len)
        score = aggregate_score(probs, target, func=agg_fuc)
        if score >= max_score:
            max_score = score
            true_false = inputs['true_false'][index]
            best_index = index
        index += 1
    return true_false, best_index


def generate_scores(model, tokenizer, inputs, agg_fuc):
    # Todo: support batch inference
    pixel_values = load_image(inputs['image_path'][0], max_num=12).to(torch.bfloat16).cuda()
    question = inputs['question'][0]
    output = []
    for i in inputs['candidates']:
        m_reason = re.search(r"\[Reasoning\](.*?)(?=\n?Answer:)", i[0], re.S)
        reasoning = m_reason.group(1) if m_reason else i[0]
        m_ans = re.search(r"Answer:\s*(.*)", i[0], re.S)
        steps = split_reasoning_block(reasoning)
        answer = m_ans.group(1).strip() if m_ans else ""
        if answer:
            steps.append(f"Answer: {answer}")
        conversation = build_conversations(question, steps)
        conv_template = get_conv_template(model.template)
        for part in conversation:
            if part['from'] == 'system':
                conv_template.system_message = part['value']
            elif part['from'] == 'human':
                conv_template.append_message(conv_template.roles[0], part['value'])
            elif part['from'] == 'gpt':
                conv_template.append_message(conv_template.roles[1], part['value'])
        prompt = conv_template.get_prompt()
        input_ids, attention_mask, image_flags, pixel_values = input_processing(model, tokenizer, prompt, pixel_values)
        output = model(pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,).logits  # (batch, seq_len, vocab_size)
        probs = torch.softmax(output, dim=-1)
        target = generate_target(input_ids, tokenizer, model.template)  # (batch, seq_len)
        score = aggregate_score(probs, target, func=agg_fuc)
        output.append(score)
    return output