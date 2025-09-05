from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import json
from transformers import AutoProcessor
from PIL import Image
import re
from utils import *
import torch


def split_step(s_id, response):
    s = f"Step {s_id}"
    s_next = f"Step {s_id+1}"
    if s_next in response:
        assistant = response.split(s_next)[0]
    elif "Final answer" in response and s in response:
        assistant = response.split("Final answer")[0]
    else:
        assistant = ""
    return assistant


def find_max_step(response):
    """
    Find the maximum step number in a response string containing steps.

    Args:
        response: String containing steps in formats like "Step 1: ...", "Step 2: ...", etc.

    Returns:
        Integer representing the highest step number found. Returns 0 if no steps are found.
    """
    # Find all occurrences of step patterns (case-insensitive)
    # Matches: "Step 1", "STEP 2", "step3", "Step: 4", etc.
    step_numbers = re.findall(r'Step[\s:]*(\d+)', response, re.IGNORECASE)

    # Return 0 if no step numbers found
    if not step_numbers:
        return 0

    # Convert found numbers from strings to integers
    step_numbers = [int(num) for num in step_numbers]

    # Return the maximum step number
    return max(step_numbers)


def read_json(source):
    with open(source, 'r', encoding='utf-8') as f:
        json_list = json.load(f)
    return json_list


def resize_image_if_needed(img, max_size=512):
    """
    Resize an image proportionally if either width or height exceeds max_size.
    Maintains the original aspect ratio while scaling down the longest side to max_size.

    :param img: PIL.Image object to be resized
    :param max_size: Maximum allowed length for the longest side (default: 512)
    :return: Resized PIL.Image object
    """
    width, height = img.size
    # Check if the longest dimension exceeds max_size
    if max(width, height) > max_size:
        # Calculate scaling ratio while maintaining aspect ratio
        scale_ratio = max_size / float(max(width, height))
        new_width = int(width * scale_ratio)
        new_height = int(height * scale_ratio)
        # Resize image using LANCZOS resampling for high quality
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return img



class MyDataset(Dataset):
    def __init__(self, data_js, max_patch_num,):
        self.data_js = data_js
        self.template = 'internvl2_5'
        self.max_patch_num = max_patch_num

    def __len__(self):
        return len(self.data_js)

    def __getitem__(self, idx):
        inputs = self.data_js[idx]
        conv_template = get_conv_template(self.template)
        for part in inputs['conversations']:
            if part['from'] == 'system':
                conv_template.system_message = part['value']
            elif part['from'] == 'human':
                conv_template.append_message(conv_template.roles[0], part['value'])
            elif part['from'] == 'gpt':
                conv_template.append_message(conv_template.roles[1], part['value'])
        prompt = conv_template.get_prompt()
        image = load_image(inputs['image'], max_num=self.max_patch_num).to(torch.bfloat16).cuda()
        id = str(inputs['id'])

        return prompt, image, id


class MyMetaDataset(Dataset):
    def __init__(self, data_js, max_patch_num):
        self.data_js = data_js
        self.template = 'internvl2_5'
        self.max_patch_num = max_patch_num

    def __len__(self):
        return len(self.data_js)

    def __getitem__(self, idx):
        inputs = self.data_js[idx]
        conv_template = get_conv_template(self.template)
        for part in inputs['conversations']:
            if part['from'] == 'system':
                conv_template.system_message = part['value']
            elif part['from'] == 'human':
                conv_template.append_message(conv_template.roles[0], part['value'])
            elif part['from'] == 'gpt':
                conv_template.append_message(conv_template.roles[1], part['value'])
        prompt = conv_template.get_prompt()
        image = load_image(inputs['image'][1:], max_num=self.max_patch_num).to(torch.bfloat16).cuda()
        label = torch.tensor(inputs["true_false"]).to(torch.bfloat16).cuda()

        return prompt, image, label

class MyTestDataset(Dataset):
    def __init__(self, data_js):
        self.data_js = data_js

    def __len__(self):
        return len(self.data_js)

    def __getitem__(self, idx):
        return self.data_js[idx]



def build_dataloader(
        train_json_file,
        meta_json_file,
        train_batch_size,
        meta_batch_size,
        max_patch_num,
):
    train_dataset = MyDataset(read_json(train_json_file), max_patch_num)
    meta_dataset = MyMetaDataset(read_json(meta_json_file), max_patch_num)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    meta_dataloader = DataLoader(meta_dataset, batch_size=meta_batch_size, shuffle=True)

    return train_dataloader, meta_dataloader


def build_test_dataloader(
        test_json_file,
):
    test_dataset = MyTestDataset(read_json(test_json_file))
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    return test_dataloader
