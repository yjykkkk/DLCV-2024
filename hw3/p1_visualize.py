import os
import json
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from transformers import CLIPVisionConfig,LlamaConfig, LlavaConfig, LlavaProcessor, CLIPImageProcessor
import time
from transformers import set_seed
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
set_seed(87) #87

def project_attention_to_image(attention_map, image_size, patch_size=14):
    height, width = image_size 
    #print("attention map min", attention_map.min(), "max", attention_map.max())
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
    patches_per_row = 24  
    attention_map = attention_map.reshape(patches_per_row, patches_per_row) 

    attention_map_tensor = torch.tensor(attention_map).float().unsqueeze(0).unsqueeze(0) 
    attention_map_tensor = torch.nn.functional.interpolate(
        attention_map_tensor, size=(height, width), mode='bilinear', align_corners=False 
    )
    attention_map_resized = attention_map_tensor.squeeze().numpy()  
    return attention_map_resized

start_time = time.time()
device =  torch.device("cpu")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16  
)
model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)

processor = AutoProcessor.from_pretrained(model_id)

image_folder = "hw3_data/p3_data/images"
output_file = "visualize.json" 

captions = {}
model = model.to(device)
output_file_path = "output.txt"
eos_token_id = processor.tokenizer.eos_token_id
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
max_length = 20

transform = transforms.Compose([
    transforms.Resize((336, 336)),
    transforms.ToTensor(),
])

for image_name in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(image_folder, image_name)
    
    raw_image = Image.open(image_path).convert("RGB") #(640, 427)
    prompt_text = "Briefly describe the image under 20 words."
    prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"#開頭\n
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(device, torch.float32)
    attention_mask = inputs["attention_mask"]
    #print("image shape", inputs['pixel_values'].shape)  #torch.Size([1, 3, 336, 336])
    #print("text shape", inputs['input_ids'].shape)  # torch.Size([1, 25])
    resize_image = transform(raw_image)
    resize_image = np.transpose(resize_image, (1, 2, 0))  # Change shape to (336, 336, 3)
    all_attention = []
    all_attention_head = [[] for _ in range(32)]
    all_word = []
    for i in range(max_length):
        attention_mask = attention_mask.to(torch.int64) 
        outputs = model.forward(
            output_attentions=True,
            input_ids=inputs['input_ids'] ,
            pixel_values=inputs['pixel_values'],
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        #print("outputs.logits shape", outputs.logits.shape)  # torch.Size([1, 600, 32064])
        next_token_logits = outputs.logits[:, -1, :] 
        next_token = torch.argmax(next_token_logits, dim=-1)
        next_word = processor.decode(next_token, skip_special_tokens=True)
        all_word.append(next_word)

        attention = outputs.attentions[-1] # which layer
        #print("attention shape", attention.shape) #torch.Size([1, 32, 600, 600])
        attention_weights = attention[0]  #([32, 600, 600])
        token_attention = attention_weights[:, -1, 1:577].mean(dim=0)  # -1 表示最後一個 token # 取圖像部分 (patches)，並對32個頭求平均
        attention_map = token_attention.detach().cpu().numpy()
        all_attention.append(attention_map)
        for i in range(32):
            head_attention_map = attention_weights[i, -1, 1:577]
            head_attention_map = head_attention_map.detach().cpu().numpy()
            all_attention_head[i].append(head_attention_map)

        if next_token.item() == eos_token_id:
            #print(f"Generated <EOS> token at step {i}. Stopping generation.")
            break  # Exit the loop if EOS is generated

        """
        fig, ax = plt.subplots()
        ax.imshow(resize_image)
        ax.imshow(attention_map, cmap='jet', alpha=0.5)  # Overlay attention with transparency
        ax.axis('off')  # Hide axes
        attention_map_filename = f"{os.path.splitext(image_name)[0]}_mean0_token_{i}.png"
        attention_map_path = os.path.join('./', attention_map_filename)
        plt.savefig(attention_map_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        """
        inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token.unsqueeze(0)], dim=1)        
        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device)], dim=1)
        generated_text = processor.decode(inputs["input_ids"][0], skip_special_tokens=True).split("ASSISTANT: ", 1)[1].strip()
        #print(f"{generated_text}")

    
    for head_idx in range(len(all_attention_head)):
        all_attention = all_attention_head[head_idx]

        cols = 5  # 每行 5 張圖
        rows = (len(all_attention)+1 + cols - 1) // cols  # 確保行數足夠容納所有圖片
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.flatten() 

        ax = axes[0]
        ax.imshow(resize_image)
        ax.set_title("<start>")
        ax.axis("off")
        for i in range(len(all_attention)):
            attention_map = all_attention[i]
            attention_map_resized = project_attention_to_image(attention_map, (resize_image.shape[0], resize_image.shape[1]), 14)
            ax = axes[i+1]
            ax.imshow(resize_image)  # 显示原图
            ax.imshow(attention_map_resized, cmap='jet', alpha=0.6)  # 将热力图叠加到原图上
            next_word = all_word[i]
            if i == len(all_attention)-1:
                next_word = "<|endoftext|>"
            #print("next_word", next_word)
            ax.set_title(f"{next_word}")
            ax.axis("off")
            
        for j in range(len(all_attention)+1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(f"{os.path.splitext(image_name)[0]}_last_head{head_idx}.png", bbox_inches='tight')
        plt.close(fig)  # 关闭当前图像，避免内存泄漏
    

    cols = 5  # 每行 5 張圖
    rows = (len(all_attention)+1 + cols - 1) // cols  # 確保行數足夠容納所有圖片
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() 

    ax = axes[0]
    ax.imshow(resize_image)
    ax.set_title("<start>")
    ax.axis("off")
    for i in range(len(all_attention)):
        attention_map = all_attention[i]
        attention_map_resized = project_attention_to_image(attention_map, (resize_image.shape[0], resize_image.shape[1]), 14)
        ax = axes[i+1]
        ax.imshow(resize_image)  
        ax.imshow(attention_map_resized, cmap='jet', alpha=0.6) 
        next_word = all_word[i]
        if i == len(all_attention)-1:
            next_word = "<|endoftext|>"
        #print("next_word", next_word)
        ax.set_title(f"{next_word}")
        ax.axis("off")
        
    for j in range(len(all_attention)+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(f"{os.path.splitext(image_name)[0]}_last_mean.png", bbox_inches='tight')
    plt.close(fig) 

with open(output_file, "w") as f:
    json.dump(captions, f, indent=4)

print(f"Captions saved to {output_file}")
end_time = time.time()
print("Execution time: ", end_time - start_time)
