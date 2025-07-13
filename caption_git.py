from transformers import pipeline, AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import time

def caption_image_pipeline(image_path):
    """Use pipeline as a high-level helper"""
    start_time = time.time()
    
    pipe = pipeline("image-to-text", model="microsoft/git-base")
    if torch.cuda.is_available():
        pipe.model = pipe.model.to("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    
    image = Image.open(image_path).convert('RGB')
    result = pipe(image)
    
    end_time = time.time()
    caption = result[0]['generated_text']
    
    return caption, end_time - start_time

def caption_image_direct(image_path):
    """Load model directly"""
    start_time = time.time()
    
    processor = AutoProcessor.from_pretrained("microsoft/git-base")
    model = AutoModelForVision2Seq.from_pretrained("microsoft/git-base")
    
    if torch.cuda.is_available():
        model = model.to("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    generated_ids = model.generate(**inputs)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    end_time = time.time()
    
    return caption, end_time - start_time

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python caption_git.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print("=== Testing Pipeline Method ===")
    caption1, time1 = caption_image_pipeline(image_path)
    print(f"Caption: {caption1}")
    print(f"Time: {time1:.3f} seconds")
    
    print("\n=== Testing Direct Model Method ===")
    caption2, time2 = caption_image_direct(image_path)
    print(f"Caption: {caption2}")
    print(f"Time: {time2:.3f} seconds")