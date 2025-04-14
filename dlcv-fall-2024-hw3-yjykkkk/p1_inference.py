import os
import json
#from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from transformers import CLIPVisionConfig,LlamaConfig, LlavaConfig, LlavaProcessor, CLIPImageProcessor
import time
from transformers import set_seed
import sys
set_seed(87) 

start_time = time.time()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_id = "llava-hf/llava-1.5-7b-hf"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16 
)

model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    quantization_config=quantization_config,
)

processor = AutoProcessor.from_pretrained(model_id)

image_folder = sys.argv[1] # "hw3_data/p1_data/images/val"
output_file = sys.argv[2] 

captions = {}

image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
for image_name in image_files:
    filename_no_ext = os.path.splitext(image_name)[0]
    
    image_path = os.path.join(image_folder, image_name)
    
    raw_image = Image.open(image_path).convert("RGB")
    prompt_text = "Briefly describe the image under 20 words."
    prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(device, torch.float16)
    output = model.generate(**inputs, 
        max_new_tokens=100,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.01
    )

    output_text = processor.decode(output[0][2:], skip_special_tokens=True)

    assistant_response = output_text.split("ASSISTANT:", 1)[1].strip()
    captions[filename_no_ext] = assistant_response

output_dir = os.path.dirname(output_file)
if output_dir: 
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir) 
with open(output_file, "w") as f:
    json.dump(captions, f, indent=4)

print(f"Captions saved to {output_file}")
end_time = time.time()
print("Execution time: ", end_time - start_time)
