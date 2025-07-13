from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

def caption_image(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    if torch.cuda.is_available():
        model = model.to("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    
    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python caption.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    caption = caption_image(image_path)
    print(f"Generated Caption: {caption}")