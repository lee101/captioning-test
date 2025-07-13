from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import time

def test_blip_optimized(image_path, num_runs=5):
    """Test BLIP with torch.compile and inference_mode"""
    print("=== BLIP Model ===")
    
    # Load model
    load_start = time.time()
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    if torch.cuda.is_available():
        model = model.to("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    
    load_time = time.time() - load_start
    print(f"Model loading time: {load_time:.3f}s")
    
    # Prepare image
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Test without optimizations
    print("\n--- Without optimizations ---")
    times = []
    for i in range(num_runs):
        start = time.time()
        with torch.no_grad():
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
        end = time.time()
        times.append(end - start)
        if i == 0:
            print(f"Caption: {caption}")
    
    avg_time = sum(times) / len(times)
    print(f"Average inference time: {avg_time:.3f}s ({num_runs} runs)")
    
    # Test with torch.compile and inference_mode
    print("\n--- With torch.compile + inference_mode ---")
    compiled_model = torch.compile(model)
    
    # Warmup
    with torch.inference_mode():
        compiled_model.generate(**inputs)
    
    times_opt = []
    for i in range(num_runs):
        start = time.time()
        with torch.inference_mode():
            out = compiled_model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
        end = time.time()
        times_opt.append(end - start)
        if i == 0:
            print(f"Caption: {caption}")
    
    avg_time_opt = sum(times_opt) / len(times_opt)
    print(f"Average inference time: {avg_time_opt:.3f}s ({num_runs} runs)")
    print(f"Speedup: {avg_time/avg_time_opt:.2f}x")
    
    return avg_time, avg_time_opt

def test_git_optimized(image_path, num_runs=5):
    """Test GitBase with torch.compile and inference_mode"""
    print("\n=== GitBase Model ===")
    
    # Load model
    load_start = time.time()
    processor = AutoProcessor.from_pretrained("microsoft/git-base")
    model = AutoModelForVision2Seq.from_pretrained("microsoft/git-base")
    
    if torch.cuda.is_available():
        model = model.to("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    
    load_time = time.time() - load_start
    print(f"Model loading time: {load_time:.3f}s")
    
    # Prepare image
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Test without optimizations
    print("\n--- Without optimizations ---")
    times = []
    for i in range(num_runs):
        start = time.time()
        with torch.no_grad():
            generated_ids = model.generate(**inputs)
            caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        end = time.time()
        times.append(end - start)
        if i == 0:
            print(f"Caption: {caption}")
    
    avg_time = sum(times) / len(times)
    print(f"Average inference time: {avg_time:.3f}s ({num_runs} runs)")
    
    # Test with torch.compile and inference_mode
    print("\n--- With torch.compile + inference_mode ---")
    compiled_model = torch.compile(model)
    
    # Warmup
    with torch.inference_mode():
        compiled_model.generate(**inputs)
    
    times_opt = []
    for i in range(num_runs):
        start = time.time()
        with torch.inference_mode():
            generated_ids = compiled_model.generate(**inputs)
            caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        end = time.time()
        times_opt.append(end - start)
        if i == 0:
            print(f"Caption: {caption}")
    
    avg_time_opt = sum(times_opt) / len(times_opt)
    print(f"Average inference time: {avg_time_opt:.3f}s ({num_runs} runs)")
    print(f"Speedup: {avg_time/avg_time_opt:.2f}x")
    
    return avg_time, avg_time_opt

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python caption_optimized.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    blip_normal, blip_opt = test_blip_optimized(image_path)
    git_normal, git_opt = test_git_optimized(image_path)
    
    print("\n=== Summary ===")
    print(f"BLIP: {blip_normal:.3f}s -> {blip_opt:.3f}s ({blip_normal/blip_opt:.2f}x speedup)")
    print(f"GitBase: {git_normal:.3f}s -> {git_opt:.3f}s ({git_normal/git_opt:.2f}x speedup)")