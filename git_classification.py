from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import time

def caption_with_classification(image_path):
    """GitBase optimized with detailed output showing classification info"""
    
    # Load optimized model
    processor = AutoProcessor.from_pretrained("microsoft/git-base")
    model = AutoModelForVision2Seq.from_pretrained(
        "microsoft/git-base", 
        torch_dtype=torch.float16
    ).to("cuda")
    
    # Compile for speed
    compiled_model = torch.compile(model)
    
    # Load and process image
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt").to("cuda")
    
    print(f"Image: {image_path}")
    print(f"Image size: {image.size}")
    print(f"Using GPU: {torch.cuda.get_device_name()}")
    
    # Warmup
    with torch.inference_mode():
        compiled_model.generate(**inputs, num_beams=1, max_length=20)
    
    # Generate caption with timing
    start_time = time.time()
    with torch.inference_mode():
        generated_ids = compiled_model.generate(
            **inputs, 
            num_beams=1, 
            max_length=20, 
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )
    end_time = time.time()
    
    # Decode caption
    caption = processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)[0]
    
    # Print results
    print(f"\nGenerated Caption: '{caption}'")
    print(f"Inference time: {(end_time - start_time) * 1000:.1f}ms")
    
    # Print token details
    print(f"\nGenerated tokens: {generated_ids.sequences[0].tolist()}")
    print(f"Number of tokens: {len(generated_ids.sequences[0])}")
    
    # Decode each token individually to show the generation process
    print("\nToken-by-token generation:")
    for i, token_id in enumerate(generated_ids.sequences[0]):
        token = processor.tokenizer.decode([token_id])
        print(f"  {i}: {token_id} -> '{token}'")
    
    # Show confidence scores if available
    if hasattr(generated_ids, 'scores') and generated_ids.scores:
        print(f"\nGeneration scores available for {len(generated_ids.scores)} steps")
        # Show top tokens for first few generation steps
        for i, scores in enumerate(generated_ids.scores[:3]):  # First 3 steps
            top_tokens = torch.topk(scores[0], 5)
            print(f"  Step {i+1} top 5 tokens:")
            for j, (score, token_id) in enumerate(zip(top_tokens.values, top_tokens.indices)):
                token = processor.tokenizer.decode([token_id])
                prob = torch.softmax(scores[0], dim=-1)[token_id].item()
                print(f"    {j+1}. '{token}' (id: {token_id}, prob: {prob:.3f})")
    
    return caption, (end_time - start_time) * 1000

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python git_classification.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    caption, inference_time = caption_with_classification(image_path)
    
    print(f"\n=== Summary ===")
    print(f"Caption: {caption}")
    print(f"Time: {inference_time:.1f}ms")