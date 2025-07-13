import torch
import time
from PIL import Image
from transformers import AutoModel, AutoProcessor

print("Loading uform-gen2-qwen-500m model...")
model = AutoModel.from_pretrained("unum-cloud/uform-gen2-qwen-500m", trust_remote_code=True, torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained("unum-cloud/uform-gen2-qwen-500m", trust_remote_code=True)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

# Load test image
try:
    image = Image.open("cat.jpg")
    print("Loaded cat.jpg")
except FileNotFoundError:
    print("cat.jpg not found, creating simple test image")
    image = Image.new('RGB', (224, 224), color='red')
    print("Created simple test image (red square)")

prompt = "Describe this image in detail"

def generate_caption(prompt, image, model, processor, device):
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    # Move inputs to same device as model
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            use_cache=True,
            max_new_tokens=256,
            eos_token_id=151645,
            pad_token_id=processor.tokenizer.pad_token_id
        )
    
    prompt_len = inputs["input_ids"].shape[1]
    decoded_text = processor.batch_decode(output[:, prompt_len:])[0]
    return decoded_text

print("\n" + "="*50)
print("FIRST RUN (cold start)")
print("="*50)
start_time = time.time()
caption1 = generate_caption(prompt, image, model, processor, device)
first_run_time = time.time() - start_time
print(f"Caption: {caption1}")
print(f"Time taken: {first_run_time:.3f} seconds")

print("\n" + "="*50)
print("SECOND RUN (warmed up)")
print("="*50)
start_time = time.time()
caption2 = generate_caption(prompt, image, model, processor, device)
second_run_time = time.time() - start_time
print(f"Caption: {caption2}")
print(f"Time taken: {second_run_time:.3f} seconds")

print(f"\nSpeedup: {first_run_time/second_run_time:.2f}x faster on second run")
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name()}")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")