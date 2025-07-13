from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import time

def benchmark_blip_optimizations(image_path, num_runs=5):
    """Test various BLIP speedup techniques"""
    
    image = Image.open(image_path).convert('RGB')
    
    results = {}
    
    # 1. Baseline (fp32)
    print("=== 1. Baseline (FP32) ===")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model = model.to("cuda")
    
    inputs = processor(image, return_tensors="pt").to("cuda")
    
    times = []
    for i in range(num_runs):
        start = time.time()
        with torch.no_grad():
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
        times.append(time.time() - start)
        if i == 0:
            print(f"Caption: {caption}")
    
    baseline_time = sum(times) / len(times)
    print(f"Average time: {baseline_time:.3f}s")
    results['baseline'] = baseline_time
    
    # 2. Mixed Precision (FP16)
    print("\n=== 2. Mixed Precision (FP16) ===")
    model_fp16 = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base", 
        torch_dtype=torch.float16
    ).to("cuda")
    
    times = []
    for i in range(num_runs):
        start = time.time()
        with torch.no_grad():
            out = model_fp16.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
        times.append(time.time() - start)
        if i == 0:
            print(f"Caption: {caption}")
    
    fp16_time = sum(times) / len(times)
    print(f"Average time: {fp16_time:.3f}s")
    print(f"Speedup vs baseline: {baseline_time/fp16_time:.2f}x")
    results['fp16'] = fp16_time
    
    # 3. BFloat16
    print("\n=== 3. BFloat16 ===")
    model_bf16 = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base", 
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    times = []
    for i in range(num_runs):
        start = time.time()
        with torch.no_grad():
            out = model_bf16.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
        times.append(time.time() - start)
        if i == 0:
            print(f"Caption: {caption}")
    
    bf16_time = sum(times) / len(times)
    print(f"Average time: {bf16_time:.3f}s")
    print(f"Speedup vs baseline: {baseline_time/bf16_time:.2f}x")
    results['bf16'] = bf16_time
    
    # 4. Reduced generation parameters
    print("\n=== 4. Faster Generation (no beam search, shorter) ===")
    times = []
    for i in range(num_runs):
        start = time.time()
        with torch.no_grad():
            out = model_fp16.generate(**inputs, num_beams=1, max_length=20, do_sample=False)
            caption = processor.decode(out[0], skip_special_tokens=True)
        times.append(time.time() - start)
        if i == 0:
            print(f"Caption: {caption}")
    
    fast_gen_time = sum(times) / len(times)
    print(f"Average time: {fast_gen_time:.3f}s")
    print(f"Speedup vs baseline: {baseline_time/fast_gen_time:.2f}x")
    results['fast_gen'] = fast_gen_time
    
    # 5. torch.compile + fp16 + fast generation
    print("\n=== 5. torch.compile + FP16 + Fast Generation ===")
    compiled_model = torch.compile(model_fp16)
    
    # Warmup
    with torch.inference_mode():
        compiled_model.generate(**inputs, num_beams=1, max_length=20)
    
    times = []
    for i in range(num_runs):
        start = time.time()
        with torch.inference_mode():
            out = compiled_model.generate(**inputs, num_beams=1, max_length=20, do_sample=False)
            caption = processor.decode(out[0], skip_special_tokens=True)
        times.append(time.time() - start)
        if i == 0:
            print(f"Caption: {caption}")
    
    compiled_time = sum(times) / len(times)
    print(f"Average time: {compiled_time:.3f}s")
    print(f"Speedup vs baseline: {baseline_time/compiled_time:.2f}x")
    results['compiled'] = compiled_time
    
    # 6. Channels last memory format
    print("\n=== 6. Channels Last Memory Format ===")
    model_channels_last = model_fp16.to(memory_format=torch.channels_last)
    inputs_cl = {k: v.to(memory_format=torch.channels_last) if v.dim() == 4 else v 
                 for k, v in inputs.items()}
    
    times = []
    for i in range(num_runs):
        start = time.time()
        with torch.no_grad():
            out = model_channels_last.generate(**inputs_cl, num_beams=1, max_length=20)
            caption = processor.decode(out[0], skip_special_tokens=True)
        times.append(time.time() - start)
        if i == 0:
            print(f"Caption: {caption}")
    
    channels_last_time = sum(times) / len(times)
    print(f"Average time: {channels_last_time:.3f}s")
    print(f"Speedup vs baseline: {baseline_time/channels_last_time:.2f}x")
    results['channels_last'] = channels_last_time
    
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python blip_speedup.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    results = benchmark_blip_optimizations(image_path)
    
    print("\n=== Final Results ===")
    baseline = results['baseline']
    for name, time_val in results.items():
        speedup = baseline / time_val
        print(f"{name:15}: {time_val:.3f}s ({speedup:.2f}x speedup)")
    
    best_name = min(results.keys(), key=lambda k: results[k])
    best_time = results[best_name]
    total_speedup = baseline / best_time
    print(f"\nBest: {best_name} with {total_speedup:.2f}x speedup!")