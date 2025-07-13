import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

# Try to disable autocast to avoid dtype issues
torch.backends.cudnn.benchmark = False

model = AutoModel.from_pretrained("unum-cloud/uform-gen2-qwen-500m", trust_remote_code=True, torch_dtype=torch.float32)
processor = AutoProcessor.from_pretrained("unum-cloud/uform-gen2-qwen-500m", trust_remote_code=True)

# Load image
try:
    image = Image.open("cat.jpg")
    print("Loaded cat.jpg")
except FileNotFoundError:
    print("cat.jpg not found")
    exit()

# Simple test
prompt = "What is in this image?"
inputs = processor(text=[prompt], images=[image], return_tensors="pt")

print("Running inference...")
with torch.no_grad():
    output = model.generate(
        **inputs,
        do_sample=False,
        use_cache=True,
        max_new_tokens=50,
        eos_token_id=151645,
        pad_token_id=processor.tokenizer.pad_token_id
    )

prompt_len = inputs["input_ids"].shape[1]
decoded_text = processor.batch_decode(output[:, prompt_len:])[0]
print(f"Caption: {decoded_text}")