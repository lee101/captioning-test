import torch
import time
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

print("Loading BLIP model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

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

def generate_caption(image, model, processor, device):
    inputs = processor(image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_length=50)
    
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

print("\n" + "="*50)
print("FIRST RUN (cold start)")
print("="*50)
start_time = time.time()
caption1 = generate_caption(image, model, processor, device)
first_run_time = time.time() - start_time
print(f"Caption: {caption1}")
print(f"Time taken: {first_run_time:.3f} seconds")

print("\n" + "="*50)
print("SECOND RUN (warmed up)")
print("="*50)
start_time = time.time()
caption2 = generate_caption(image, model, processor, device)
second_run_time = time.time() - start_time
print(f"Caption: {caption2}")
print(f"Time taken: {second_run_time:.3f} seconds")

print(f"\nSpeedup: {first_run_time/second_run_time:.2f}x faster on second run")
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name()}")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")