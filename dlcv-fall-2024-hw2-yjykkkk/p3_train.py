# reference: https://github.com/rinongal/textual_inversion
import argparse
import itertools
import math
import os
import random
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from torch import nn
import PIL
from torch.utils.data import ConcatDataset
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTokenizer
from ldm.modules.encoders.modules import FrozenCLIPEmbedder
from ldm.modules.diffusionmodules.openaimodel import UNetModel
import requests
import glob
from io import BytesIO
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

initial_token1 = "dog" 
initial_token2 = "cartoon"
placeholder_token_1 = "<new1>"
placeholder_token_2 = "<new2>"

imagenet_style_templates_small = [ 
    "an expressive painting in the style of {}",
    "an impressionist rendering in the style of {}",
    "a vibrant painting with bold colors in the style of {}",
    "a softly lit painting in the style of {}",
    "an abstract artwork in the style of {}",
    "a textured artwork in the style of {}",
    "a detailed close-up in the style of {}",
    "a vintage painting in the style of {}",
    "a high-contrast painting in the style of {}",
    "a sepia-toned picture in the style of {}",
    "a painting with muted tones in the style of {}",
    "a surrealist artwork in the style of {}",
    "a vibrant, colorful piece in the style of {}",
    "an intricate painting in the style of {}",
    "a minimalist rendition in the style of {}",
    "a piece with soft shadows in the style of {}",
    "a dreamlike painting in the style of {}",
    "an avant-garde painting in the style of {}",
    "a digital painting inspired by {}",
    "an artistic depiction in the style of {}",
]
imagenet_templates_small = [ #20
    "a beautifully detailed photo of a {}",
    "a studio shot of a {}",
    "a natural light photo of a {}",
    "a high-resolution close-up of the {}",
    "an artistic photo of a {}",
    "a soft-focus photo of the {}",
    "a vibrant, colorful photo of a {}",
    "a portrait-style photo of the {}",
    "a dynamic angle shot of the {}",
    "a photo with dramatic lighting of a {}",
    "a candid photo of the {}",
    "a landscape shot of a {}",
    "a well-composed photo of the {}",
    "a macro shot of the {}",
    "a beautiful photo of one {}",
    "a richly colored photo of a {}",
    "a monochrome photo of a {}",
    "a creative rendering of a {}",
    "a dreamy photo of the {}",
    "a professionally lit photo of a {}",
]


class StyleDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_root,
        learnable_property,  # ["object", "style"]
        placeholder_token,  # <new1>, <new2>
        size=512,
        repeats=20,
        flip_p=0.5,
        set="train",
        center_crop=False,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)] * repeats
        self.num_images = len(self.image_paths)
        self.templates = imagenet_style_templates_small 
        self._length = len(self.image_paths)
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # 50% 概率水平翻转
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0))], p=0.3),  # 30% 概率添加轻微模糊
            transforms.RandomRotation(degrees=(-10, 10)),  # 随机旋转 ±10 度
        ])


    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image_path = self.image_paths[i]
        template = self.templates[i//5]
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        placeholder_string = self.placeholder_token
        text = template.format(placeholder_string)
        example["input_ids"] = self.tokenizer(
            text, 
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )["input_ids"]
        example["text"] = text 
        image = image.resize((self.size, self.size)) 
        image = self.transform(image)  

        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        example["pixel_values"] = torch.FloatTensor(np.transpose(image, (2, 0, 1)))
        return example

class ObjectDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_root,
        learnable_property,  # ["object", "style"]
        placeholder_token,  # <new1>, <new2>
        size=512,
        repeats=20,
        flip_p=0.5,
        set="train",
        center_crop=False,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)] * repeats
        self.num_images = len(self.image_paths)
        self.templates = imagenet_templates_small
        self._length = len(self.image_paths)
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.GaussianBlur((3, 3), (0.1, 1.0))], p=0.2),
        ])
    def __len__(self):
        return self._length
    def __getitem__(self, i):
        example = {}
        image_path = self.image_paths[i]
        template = self.templates[i//5]
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        placeholder_string = self.placeholder_token
        text = template.format(placeholder_string)
        example["input_ids"] = self.tokenizer(
            text, 
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )["input_ids"]
        example["text"] = text  # 保存生成的文本字符串
        image = image.resize((self.size, self.size))  # 调整大小
        image = self.transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        example["pixel_values"] = torch.FloatTensor(np.transpose(image, (2, 0, 1)))
        return example

def create_dataloader(train_dataset, train_batch_size):
    return torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(string, truncation=True, max_length=77, return_length=True,
                               return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"]
    num_tokens = torch.count_nonzero(tokens - 49407)  # 计算非填充 token 的数量
    if num_tokens > 2:
        print(f"String '{string}' maps to {num_tokens} tokens.")
    return tokens[0, 1:num_tokens]  # 返回所有非填充的token


def get_embedding_for_clip_token(embedder, token):
    return embedder(token.unsqueeze(0))[0, 0]

class EmbeddingManager(nn.Module):
    def __init__(
            self,
            embedder,
            placeholder_strings=["<new1>", "<new2>"],
            initializer_words=["dog", "cartoon"],
            per_image_tokens=False,
            num_vectors_per_token=1,
            progressive_words=False,
            **kwargs
    ):
        super().__init__()
        self.string_to_token_dict = {}
        self.string_to_param_dict = nn.ParameterDict()
        self.initial_embeddings = nn.ParameterDict()  # These should not be optimized
        self.progressive_words = progressive_words
        self.progressive_counter = 0
        self.max_vectors_per_token = num_vectors_per_token
        if hasattr(embedder, 'tokenizer'):  
            self.is_clip = True
            get_token_for_string = partial(get_clip_token_for_string, embedder.tokenizer)
            get_embedding_for_tkn = partial(get_embedding_for_clip_token, embedder.transformer.text_model.embeddings)
            token_dim = 768  

        for idx, placeholder_string in enumerate(placeholder_strings):
            token = get_token_for_string(placeholder_string) 
            if initializer_words and idx < len(initializer_words):
                init_word_token = get_token_for_string(initializer_words[idx])
                init_word_token = init_word_token.to(embedder.device)
                with torch.no_grad():
                    init_word_embedding = get_embedding_for_tkn(init_word_token.to("cuda"))
                token_params = torch.nn.Parameter(init_word_embedding.unsqueeze(0).repeat(num_vectors_per_token, 1), requires_grad=True)
                self.initial_embeddings[placeholder_string] = torch.nn.Parameter(init_word_embedding.unsqueeze(0).repeat(num_vectors_per_token, 1), requires_grad=False)
            else:
                token_params = torch.nn.Parameter(torch.rand(size=(num_vectors_per_token, token_dim)), requires_grad=True)
            self.string_to_token_dict[placeholder_string] = token  
            self.string_to_param_dict[placeholder_string] = token_params
            

    def forward(self, tokenized_text, embedded_text):
        embedded_text = embedded_text.to(tokenized_text.device)

        for placeholder_string, token_id in self.string_to_token_dict.items():
            placeholder_embedding = self.string_to_param_dict[placeholder_string].to(tokenized_text.device)
            mask = (tokenized_text == token_id)  
            embedded_text = torch.where(mask.unsqueeze(-1), placeholder_embedding, embedded_text)

        return embedded_text

class CustomCLIPEmbedder(FrozenCLIPEmbedder):
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__(version, device, max_length)
        self.device = device 
        self.to(device) 
        self.new_token_ids = []

    def add_placeholder_token(self, placeholder_token):
        num_added_tokens = self.tokenizer.add_tokens([placeholder_token])  # 确保以列表形式添加
        if num_added_tokens == 0:
            raise ValueError(f"The tokenizer already contains the token {placeholder_token}.")
        new_token_id = self.tokenizer.convert_tokens_to_ids(placeholder_token)
        self.new_token_ids.append(new_token_id)  # Track this token ID
        print(f"Added {num_added_tokens} tokens. New token ID: {new_token_id}")
        self.resize_embeddings()

    def resize_embeddings(self):
        old_embedding_weight = self.transformer.get_input_embeddings().weight.data
        new_size = len(self.tokenizer)  # 更新为当前 tokenizer 的大小
        new_embedding_weight = torch.nn.Embedding(new_size, old_embedding_weight.size(1)).to(self.device)
        
        new_embedding_weight.weight.data[:old_embedding_weight.size(0)] = old_embedding_weight
        self.transformer.set_input_embeddings(new_embedding_weight)
        print(f"Resized embeddings to {new_size} tokens.")

    def set_placeholder_embedding(self, initializer_token_id):
        token_embeds = self.transformer.get_input_embeddings().weight.data
        new_token_id = self.new_token_ids[-1]  # 获取最新添加的 token ID
        
        # 设置新 token 的嵌入
        if new_token_id < token_embeds.size(0) and initializer_token_id < token_embeds.size(0):
            token_embeds[new_token_id] = token_embeds[initializer_token_id]
            token_embeds[new_token_id].requires_grad = False 
        else:
            raise IndexError(f"Initializer token ID {initializer_token_id} or new token ID {new_token_id} is out of bounds.")

if __name__ == "__main__":
    config = OmegaConf.load("./stable-diffusion/configs/stable-diffusion/v1-inference.yaml")
    ldm = load_model_from_config(config, "./stable-diffusion/ldm/models/stable-diffusion-v1/model.ckpt")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ldm = ldm.to(device)
    
    frozen_clip_embedder = ldm.cond_stage_model
    text_encoder = CustomCLIPEmbedder()
    text_encoder = text_encoder.to(device)
    text_encoder.load_state_dict(frozen_clip_embedder.state_dict())
    tokenizer = text_encoder.tokenizer

    text_encoder.add_placeholder_token("<new1>")
    text_encoder.add_placeholder_token("<new2>")
    
    # Initialize first token
    token_ids_1 = tokenizer.encode(initial_token1, add_special_tokens=False)
    if len(token_ids_1) > 1:
        raise ValueError("The initializer token must be a single token.")
    initializer_token_id_1 = token_ids_1[0]
    print("initializer_token_id_1: ", initializer_token_id_1)
    text_encoder.set_placeholder_embedding(initializer_token_id_1)
    # Initialize second token
    token_ids_2 = tokenizer.encode(initial_token2, add_special_tokens=False)
    if len(token_ids_2) > 1:
        raise ValueError("The initializer token must be a single token.")
    initializer_token_id_2 = token_ids_2[0]
    print("initializer_token_id_2: ", initializer_token_id_2)
    text_encoder.set_placeholder_embedding(initializer_token_id_2)
    
    # Get placeholder token IDs
    placeholder_token_id_1 = tokenizer.convert_tokens_to_ids("<new1>")
    placeholder_token_id_2 = tokenizer.convert_tokens_to_ids("<new2>")
    print(f"Placeholder token IDs: {placeholder_token_id_1}, {placeholder_token_id_2}")

    placeholder_token_id_1 = text_encoder.new_token_ids[0]
    placeholder_token_id_2 = text_encoder.new_token_ids[1]
    print(f"Placeholder tokens: {placeholder_token_1}, {placeholder_token_2}")
    print(f"Placeholder token IDs: {placeholder_token_id_1}, {placeholder_token_id_2}")

    encoded_id_1 = tokenizer.encode("<new1>", add_special_tokens=False)
    print(f"Encoded IDs for <new1>: {encoded_id_1}")
    encoded_id_2 = tokenizer.encode("<new2>", add_special_tokens=False)
    print(f"Encoded IDs for <new2>: {encoded_id_2}")
    

    # Set embeddings for both tokens
    token_embeds = text_encoder.transformer.get_input_embeddings().weight.data
    token_embeds[placeholder_token_id_1] = token_embeds[initializer_token_id_1]
    token_embeds[placeholder_token_id_2] = token_embeds[initializer_token_id_2]
    
    embedding_manager = EmbeddingManager(
        embedder=text_encoder,
        placeholder_strings=['<new1>', '<new2>'],  # 占位符 token 列表
        initializer_words=['dog', 'cartoon'], 
        num_vectors_per_token=1,  # 每个 token 一个嵌入向量
    )

    object_dataset = ObjectDataset(
        data_root="./hw2_data/textual_inversion/0/",
        tokenizer=tokenizer,
        size=512,
        placeholder_token="<new1>",
        repeats=20,
        learnable_property="object", #object, style
        center_crop=False,
        set="train",
    )
    style_dataset = StyleDataset(
        data_root="./hw2_data/textual_inversion/1/",
        tokenizer=tokenizer,
        size=512,
        placeholder_token="<new2>",
        repeats=20,
        learnable_property="style", #object, style
        center_crop=False,
        set="train",
    )
    train_dataset = ConcatDataset([object_dataset, style_dataset])

    logger = print
    train_batch_size = 4 ##22000
    gradient_accumulation_steps = 1
    #learning_rate = 5e-3 #1e-
    max_train_steps = 100000 #100 epoch
    output_dir = "p3_embed/" 
    os.makedirs(output_dir, exist_ok=True)

    train_dataloader = create_dataloader(train_dataset, train_batch_size)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    num_train_epochs = 50
    logger("***** Running training *****")
    logger(f"  Num examples = {len(train_dataset)}")
    logger(f"  Batch size = {train_batch_size}")
    ldm.first_stage_model.eval()
    
    embedding_layer = text_encoder.transformer.get_input_embeddings()

    param_new1 = embedding_manager.string_to_param_dict["<new1>"]
    param_new2 = embedding_manager.string_to_param_dict["<new2>"]
    optimizer = torch.optim.AdamW(
        [
            {"params": param_new1, "lr": 5e-3},
            {"params": param_new2, "lr": 3e-3}
        ]
    )

    for param in embedding_manager.parameters():
        print("grad: ", param.requires_grad)  #true
    for name, param in embedding_manager.named_parameters():
        if param.requires_grad:
            print(f"Parameter Name: {name}")
            print(f"Parameter Value Shape: {param.data.shape}")
    embedding_manager.to(device)
    embedding_manager.train()
    print("grad1", embedding_manager.string_to_param_dict["<new1>"].requires_grad)
    print("grad2", embedding_manager.string_to_param_dict["<new2>"].requires_grad)
    print(f"Embedding weight shape: {embedding_layer.weight.shape}") # torch.Size([49410, 768])
    param_new1.requires_grad = True
    param_new2.requires_grad = True

    for epoch in range(num_train_epochs):
        epoch_loss = 0.0
        num_batches = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            optimizer.zero_grad()
            pixel_values = batch['pixel_values'].to('cuda')
            text = batch['text']  # 从 batch 中获取文本字符串
            input_ids = batch['input_ids'].squeeze(1)
            with torch.no_grad():
                latents = ldm.encode_first_stage(pixel_values).sample()
            latents *= 0.18215
            text_embedding = text_encoder(text) #here???
            text_embedding = embedding_manager(input_ids, text_embedding)
            text_embedding = text_embedding.to(device)

            loss = ldm(latents, text_embedding)[0]
            epoch_loss += loss.item()
            num_batches += 1

            loss.backward()
            optimizer.step()
            
        average_loss = epoch_loss / num_batches
        print(f"Epoch [{epoch+1}/{num_train_epochs}], Loss: {average_loss:.4f}")

    # Save final model embeddings after training
        param_new1 = embedding_manager.string_to_param_dict["<new1>"]
        param_new2 = embedding_manager.string_to_param_dict["<new2>"]
        output_path1 = os.path.join(output_dir, f"new1_{epoch}.bin")
        output_path2 = os.path.join(output_dir, f"new2_{epoch}.bin")
        torch.save(param_new1.data, output_path1)
        torch.save(param_new2.data, output_path2)

    logger(f"Model and embeddings saved at {output_path}")

