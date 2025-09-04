from transformers import AutoModel
import torch.nn.functional as F
from utils import *
import torch
import torch.nn as nn

# # Define LoRA configuration
# lora_config = LoraConfig(
#     r=8,             # Rank for dimensionality reduction (higher = better performance but more compute)
#     lora_alpha=16,   # Scaling factor for LoRA weights
#     target_modules=["q_proj", "v_proj"],  # Modules to apply LoRA to (GPT example)
#     lora_dropout=0.1,  # Dropout probability for LoRA layers
#     bias="none"      # Whether to apply LoRA to biases ("none", "all", or "lora_only")
# )

def generate_steps_with_logit(
        self,
        tokenizer,
        prompt,
        pixel_values,
        num_patches_list=None,
        max_steps=None,
        IMG_START_TOKEN='<img>',
        IMG_END_TOKEN='</img>',
        IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
        PLACEHOLDER=None,
        str2score=None,
):
    # TODO: support batch inference
    if str2score is None:
        str2score = {'+': 1, '-': 0}

    if PLACEHOLDER is None:
        PLACEHOLDER = '+'
        PLACEHOLDER_2 = '-'

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

    candidate_tokens = []
    candidate_weights = []

    # Prepare Query
    for k, v in str2score.items():
        k_id = tokenizer.convert_tokens_to_ids(k)
        assert k_id != tokenizer.unk_token_id

        candidate_tokens.append(k_id)
        candidate_weights.append(v)

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

    # Forward
    idx = find_placeholder_idx(self.template, tokenizer, input_ids, PLACEHOLDER=PLACEHOLDER)
    idx_2 = find_placeholder_idx(self.template, tokenizer, input_ids, PLACEHOLDER=PLACEHOLDER_2)
    logits = self(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_flags=image_flags,
    ).logits
    logits_1 = logits[0][idx, :].softmax(dim=-1)
    logits_2 = logits[0][idx_2, :].softmax(dim=-1)
    gt_1 = torch.zeros(logits_1.shape,
                        dtype=torch.bfloat16,
                        device=device)
    gt_1[:, candidate_tokens[0]] = 1
    gt_2 = torch.zeros(logits_2.shape,
                       dtype=torch.bfloat16,
                       device=device
                       )
    gt_2[:, candidate_tokens[1]] = 1
    output_logits = torch.cat((logits_1, logits_2), dim=0)
    gt = torch.cat((gt_1, gt_2), dim=0)
    # print(gt[:, candidate_tokens[0]])
    # print(output_logits[:, candidate_tokens[0]])
    # print(gt[:, candidate_tokens[1]])
    # print(output_logits[:, candidate_tokens[1]])

    return (output_logits, gt)

def compute_supervised_loss(logits, target):
    """
    logits: [batch, seq_len, vocab]
    target: [batch, seq_len], with -1 indicating positions not supervised
    """
    position_mask = (target != -1)

    logits_flat = logits.view(-1, logits.size(-1))  # [B*T, V]
    target_flat = target.view(-1)
    mask_flat = position_mask.view(-1)

    logits_supervised = logits_flat[mask_flat]
    target_supervised = target_flat[mask_flat]

    if target_supervised.numel() == 0:
        return torch.tensor(0.0, requires_grad=True)  # 没有监督位置

    loss = F.cross_entropy(logits_supervised, target_supervised)
    return loss

def print_target_token_logits(logits: torch.Tensor, target: torch.Tensor):
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

    # 打印信息
    print("📋 Target Token Logits:")
    for b, s, t, l in zip(batch_indices, seq_indices, token_indices, supervised_logits):
        print(f"[batch {b.item()}][pos {s.item()}] → target token: {t.item()}, logit: {l.item():.4f}")

class DomainTable(nn.Module):
    def __init__(self, domain_to_idx):
        """
        Args:
            domain_to_idx (dict):
                Mapping from domain strings to integer indices, e.g., {"domain_a": 0, "domain_b": 1}.
        """
        super(DomainTable, self).__init__()
        self.domain_to_idx = domain_to_idx
        self.num_domains = len(domain_to_idx)

        # Create learnable raw weights (initialized to zero)
        self.raw_weights = nn.Parameter(torch.zeros(self.num_domains))

    def forward(self, domain_strings, x):
        """
        Args:
            domain_strings (list[str] or tuple[str]):
                Domain names for each sample in the batch. Length should match x's batch_size.
            x (torch.Tensor):
                Input tensor of shape (batch_size, 1), containing a single value per sample.

        Returns:
            torch.Tensor:
                Output tensor of same shape (batch_size, 1), where each element is the original input
                multiplied by its corresponding domain weight.
        """
        # Apply softplus to ensure weights are positive
        positive_weights = torch.nn.functional.softplus(self.raw_weights)

        # Normalize weights by their mean to maintain scale
        mean_weights = positive_weights.mean()
        normalized_weights = positive_weights / mean_weights

        # Convert domain strings to indices matching batch order
        idxes = [self.domain_to_idx[d] for d in domain_strings]
        idxes = torch.tensor(idxes, dtype=torch.long, device=x.device)  # [batch_size]

        # Retrieve domain weights for each sample in the batch [batch_size]
        domain_weights = normalized_weights[idxes]

        # Reshape weights to match input tensor dimensions [batch_size, 1]
        domain_weights = domain_weights.view(-1, 1)

        # Element-wise multiplication: each input value multiplied by its domain weight
        out = x * domain_weights
        return out


class InstanceTable(nn.Module):
    def __init__(self, domain_to_idx, activation_function, initialization):
        """
        Args:
            domain_to_idx (dict):
                Mapping from domain strings to integer indices, e.g., {"domain_a": 0, "domain_b": 1}.
        """
        super(InstanceTable, self).__init__()
        self.domain_to_idx = domain_to_idx
        self.num_domains = len(domain_to_idx)

        self.raw_weights = nn.Parameter(torch.ones(self.num_domains) * initialization)

        if activation_function == 'ReLU':
            self.relu = torch.nn.ReLU()
        elif activation_function == 'LeakyReLU':
            self.relu = torch.nn.LeakyReLU()
        elif activation_function == 'No':
            self.relu = torch.nn.Identity()
        elif activation_function == 'Clip':
            self.relu = lambda t: torch.clamp(t, min=-1.0, max=3.0)

    def forward(self, domain_strings, x):
        """
        Args:
            domain_strings (list[str] or tuple[str]):
                Domain names for each sample in the batch. Length should match x's batch_size.
            x (torch.Tensor):
                Input tensor of shape (batch_size, 1), containing a single value per sample.

        Returns:
            torch.Tensor:
                Output tensor of same shape (batch_size, 1), where each element is the original input
                multiplied by its corresponding domain weight.
        """
        positive_weights = self.relu(self.raw_weights)

        idxes = [self.domain_to_idx[d] for d in domain_strings]
        idxes = torch.tensor(idxes, dtype=torch.long, device=x.device)  # [batch_size]

        domain_weights = positive_weights[idxes]

        domain_weights = domain_weights.view(-1, 1)

        out = x * domain_weights
        return out


class HiddenLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(HiddenLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


class InstanceNet(nn.Module):
    def __init__(self, activation_function, hidden_size=500, num_layers=1):
        super(InstanceNet, self).__init__()
        self.first_hidden_layer = HiddenLayer(2, hidden_size)
        self.rest_hidden_layers = nn.Sequential(
            *[HiddenLayer(hidden_size, hidden_size) for _ in range(num_layers - 1)]
        )
        self.output_layer = nn.Linear(hidden_size, 1)

        if activation_function == 'ReLU':
            self.relu = torch.nn.ReLU()
        elif activation_function == 'LeakyReLU':
            self.relu = torch.nn.LeakyReLU()
        elif activation_function == 'No':
            self.relu = torch.nn.Identity()
        elif activation_function == 'Clip':
            self.relu = lambda t: torch.clamp(t, min=-1.0, max=3.0)
        elif activation_function == 'sigmoid':
            self.relu = torch.nn.Sigmoid()

    def forward(self, x, target):
        # 找到 target 中等于 10 (+) 或 12 (-) 的位置
        batch_indices, seq_indices = torch.nonzero((target == 10) | (target == 12), as_tuple=True)
        # 从这些位置取出 token=10 和 token=12 的 logits
        logits_10 = x[batch_indices, seq_indices, 10]  # 对应 token 10
        logits_12 = x[batch_indices, seq_indices, 12]  # 对应 token 12
        x = torch.stack([logits_10, logits_12], dim=1)  # [num_positions, 2]
        x = torch.softmax(x, dim=1)
        x = self.first_hidden_layer(x)
        x = self.rest_hidden_layers(x)
        x = self.output_layer(x)
        x = torch.mean(x, dim=0)
        return self.relu(x)