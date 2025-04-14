import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from ldm.modules.encoders.modules import FrozenCLIPEmbedder
from ldm.util import instantiate_from_config
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from torch import nn
from transformers import AutoFeatureExtractor
import json
from functools import partial
import sys

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
            initializer_words=["dog", "David Revoy"],
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
        if hasattr(embedder, 'tokenizer'):  # 使用Stable Diffusion的CLIP编码器
            self.is_clip = True
            get_token_for_string = partial(get_clip_token_for_string, embedder.tokenizer)
            get_embedding_for_tkn = partial(get_embedding_for_clip_token, embedder.transformer.text_model.embeddings)
            token_dim = 768  # 根据模型的 token 维度来设置

        for idx, placeholder_string in enumerate(placeholder_strings):
            token = get_token_for_string(placeholder_string)  # 只获取单一 token
            if initializer_words and idx < len(initializer_words):
                init_word_token = get_token_for_string(initializer_words[idx])
                init_word_token = init_word_token.to(embedder.device)
                with torch.no_grad():
                    init_word_embedding = get_embedding_for_tkn(init_word_token.to(embedder.device)) #"cuda" #here
                token_params = torch.nn.Parameter(init_word_embedding.unsqueeze(0).repeat(num_vectors_per_token, 1), requires_grad=True)
                self.initial_embeddings[placeholder_string] = torch.nn.Parameter(init_word_embedding.unsqueeze(0).repeat(num_vectors_per_token, 1), requires_grad=False)
            else:
                token_params = torch.nn.Parameter(torch.rand(size=(num_vectors_per_token, token_dim)), requires_grad=True)
            self.string_to_token_dict[placeholder_string] = token  # 单一 token ID
            self.string_to_param_dict[placeholder_string] = token_params

    def forward(self, tokenized_text, embedded_text):
        embedded_text = embedded_text.to(tokenized_text.device)

        for placeholder_string, token_id in self.string_to_token_dict.items():
            placeholder_embedding = self.string_to_param_dict[placeholder_string].to(tokenized_text.device)
            mask = (tokenized_text == token_id)  # 生成 mask
            embedded_text = torch.where(mask.unsqueeze(-1), placeholder_embedding, embedded_text)

        return embedded_text

class CustomCLIPEmbedder(FrozenCLIPEmbedder):
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__(version, device, max_length)
        self.device = device 
        self.to(device)  
        self.new_token_ids = []

    def add_placeholder_token(self, placeholder_token):
        num_added_tokens = self.tokenizer.add_tokens([placeholder_token])  
        if num_added_tokens == 0:
            raise ValueError(f"The tokenizer already contains the token {placeholder_token}.")
        new_token_id = self.tokenizer.convert_tokens_to_ids(placeholder_token)
        self.new_token_ids.append(new_token_id)  # Track this token ID
        self.resize_embeddings()

    def resize_embeddings(self):
        old_embedding_weight = self.transformer.get_input_embeddings().weight.data
        new_size = len(self.tokenizer)  
        new_embedding_weight = torch.nn.Embedding(new_size, old_embedding_weight.size(1)).to(self.device)
        new_embedding_weight.weight.data[:old_embedding_weight.size(0)] = old_embedding_weight
        self.transformer.set_input_embeddings(new_embedding_weight)

    def set_placeholder_embedding(self, initializer_token_id):
        token_embeds = self.transformer.get_input_embeddings().weight.data
        new_token_id = self.new_token_ids[-1]  
        if new_token_id < token_embeds.size(0) and initializer_token_id < token_embeds.size(0):
            token_embeds[new_token_id] = token_embeds[initializer_token_id]
        else:
            raise IndexError(f"Initializer token ID {initializer_token_id} or new token ID {new_token_id} is out of bounds.")

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


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

    #model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x

if __name__ == "__main__":
    tic = time.time()
    ddim_steps = 50
    seed_everything(42)
    config = OmegaConf.load("stable-diffusion/configs/stable-diffusion/v1-inference.yaml")
    ckpt_path = sys.argv[3]  #"stable-diffusion/ldm/models/stable-diffusion-v1/model.ckpt"
    model = load_model_from_config(config, ckpt_path)
    frozen_clip_embedder = model.cond_stage_model
    text_encoder = CustomCLIPEmbedder()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    text_encoder = text_encoder.to(device)
    text_encoder.load_state_dict(frozen_clip_embedder.state_dict())
    tokenizer = text_encoder.tokenizer
    text_encoder.add_placeholder_token("<new1>")
    text_encoder.add_placeholder_token("<new2>")

    embedding_manager = EmbeddingManager(
        embedder=text_encoder,
        placeholder_strings=['<new1>', '<new2>'],  # 占位符 token 列表
        initializer_words=['dog', 'cartoon'], 
        num_vectors_per_token=1,  # 每个 token 一个嵌入向量
    )
    
    param_new1_loaded = torch.nn.Parameter(torch.load("best_new1.bin"))
    embedding_manager.string_to_param_dict["<new1>"] = param_new1_loaded
    param_new2_loaded = torch.nn.Parameter(torch.load("best_new2.bin"))
    embedding_manager.string_to_param_dict["<new2>"] = param_new2_loaded
    
    sampler = DPMSolverSampler(model)
    outdir = sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    outpath = outdir

    #Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    n_samples = 1
    batch_size = n_samples
    scale = 7.5
    json_file = sys.argv[1]
    prompts = {}
    with open(json_file, 'r') as f:
        prompt_data = json.load(f)   

    start_code = None
    
    precision_scope = autocast

    n_iter = 25
    f = 8 # downsampling factor
    C = 4 # latent channels
    concept_num = len(prompt_data)
    for concept_id in range(concept_num):
        concept_path = os.path.join(outdir, str(concept_id))
        os.makedirs(concept_path, exist_ok=True)
        prompt_list = prompt_data[str(concept_id)]['prompt']
        prompt_num = len(prompt_list)
        for prompt_id in range(prompt_num):
            sample_path = os.path.join(concept_path, str(prompt_id))
            os.makedirs(sample_path, exist_ok=True)
            prompt = prompt_list[prompt_id]
            print("prompt", prompt)
            #print("concept path: ", concept_path)
            #print("sample path: ", sample_path)
            base_count = len(os.listdir(sample_path))
            data = [batch_size * [prompt]]

            with torch.no_grad():
                with precision_scope("cuda"):
                    with model.ema_scope():
                        for n in range(n_iter):
                            for prompts in data:
                                uc = None
                                if scale != 1.0:
                                    uc = model.get_learned_conditioning(batch_size * [""])
                                if isinstance(prompts, tuple):
                                    prompts = list(prompts)
                                tokenized_prompt = tokenizer(
                                    prompts, 
                                    padding="max_length",
                                    truncation=True,
                                    max_length=77,
                                    return_tensors="pt",
                                )["input_ids"]
                                text_embedding = text_encoder(prompts)
                                text_embedding = embedding_manager(tokenized_prompt, text_embedding).to(device)
                                shape = [C, 512 // f, 512 // f]
                                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                                conditioning=text_embedding,
                                                                batch_size=n_samples,
                                                                shape=shape,
                                                                verbose=False,
                                                                unconditional_guidance_scale=scale,
                                                                unconditional_conditioning=uc,
                                                                eta=0.0,
                                                                x_T=start_code)

                                x_samples_ddim = model.decode_first_stage(samples_ddim)
                                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                                x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                                
                                for x_sample in x_checked_image_torch:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    img = Image.fromarray(x_sample.astype(np.uint8))
                                    img = put_watermark(img, wm_encoder)
                                    img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                                    base_count += 1
    toc = time.time()
    print("time: ", toc - tic)