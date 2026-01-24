#!/usr/bin/env python3
"""
Convert fine-tuned TinyLlama model to GGUF format for llama.cpp
===============================================================

This script:
1. Merges the LoRA adapter with the base model
2. Saves the merged model
3. Converts to GGUF format using llama.cpp

Requirements:
    pip install torch transformers peft

For GGUF conversion, you need llama.cpp:
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    pip install -r requirements.txt
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FINETUNED_PATH = "./tinyllama-mental-health-finetuned"
MERGED_OUTPUT_PATH = "./tinyllama-mental-health-merged"
GGUF_OUTPUT_PATH = "./models/tinyllama-mental-health-q4_k_m.gguf"


def merge_lora_weights(base_model_name: str, adapter_path: str, output_path: str):
    """
    Merge LoRA adapter weights with base model and save.
    """
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",  # Use CPU for merging to avoid memory issues
        trust_remote_code=True,
    )
    
    print(f"Loading tokenizer from: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    print(f"Loading LoRA adapter from: {adapter_path}")
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print("Merging LoRA weights with base model...")
        model = model.merge_and_unload()
    except Exception as e:
        print(f"Note: Could not load as PEFT model ({e}), trying as regular model...")
        model = AutoModelForCausalLM.from_pretrained(
            adapter_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )
    
    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    print("Merge complete!")
    return output_path


def convert_to_gguf(merged_model_path: str, output_gguf_path: str, quantization: str = "q4_k_m"):
    """
    Convert merged model to GGUF format using llama.cpp.
    
    Quantization options:
        - q4_0: 4-bit quantization (smallest, fastest)
        - q4_k_m: 4-bit K-quant medium (good balance)
        - q5_k_m: 5-bit K-quant medium (better quality)
        - q8_0: 8-bit quantization (best quality, larger)
        - f16: 16-bit float (no quantization)
    """
    import subprocess
    
    # Create output directory
    os.makedirs(os.path.dirname(output_gguf_path), exist_ok=True)
    
    # Path to llama.cpp convert script
    llama_cpp_path = os.environ.get("LLAMA_CPP_PATH", "./llama.cpp")
    convert_script = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")
    quantize_binary = os.path.join(llama_cpp_path, "build", "bin", "llama-quantize")
    
    if not os.path.exists(convert_script):
        print(f"""
ERROR: llama.cpp not found at {llama_cpp_path}

Please install llama.cpp:
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    pip install -r requirements.txt
    
    # Build quantize tool (optional, for quantization)
    mkdir build && cd build
    cmake ..
    cmake --build . --config Release
    
Then set LLAMA_CPP_PATH environment variable or place llama.cpp in current directory.
""")
        return None
    
    # Step 1: Convert to GGUF (f16)
    f16_gguf_path = output_gguf_path.replace(".gguf", "-f16.gguf")
    print(f"Converting to GGUF (f16): {f16_gguf_path}")
    
    cmd = [
        "python", convert_script,
        merged_model_path,
        "--outfile", f16_gguf_path,
        "--outtype", "f16"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error during conversion: {result.stderr}")
        return None
    
    print("F16 GGUF created successfully!")
    
    # Step 2: Quantize (if requested)
    if quantization != "f16":
        if not os.path.exists(quantize_binary):
            print(f"Quantize binary not found at {quantize_binary}")
            print(f"Using f16 model: {f16_gguf_path}")
            return f16_gguf_path
        
        print(f"Quantizing to {quantization}: {output_gguf_path}")
        cmd = [quantize_binary, f16_gguf_path, output_gguf_path, quantization]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error during quantization: {result.stderr}")
            return f16_gguf_path
        
        # Remove f16 intermediate file
        os.remove(f16_gguf_path)
        print(f"Quantized GGUF created: {output_gguf_path}")
        return output_gguf_path
    
    return f16_gguf_path


def main():
    parser = argparse.ArgumentParser(description="Convert fine-tuned model to GGUF")
    parser.add_argument("--base-model", default=BASE_MODEL, help="Base model name/path")
    parser.add_argument("--adapter-path", default=FINETUNED_PATH, help="Fine-tuned adapter path")
    parser.add_argument("--merged-output", default=MERGED_OUTPUT_PATH, help="Merged model output path")
    parser.add_argument("--gguf-output", default=GGUF_OUTPUT_PATH, help="GGUF output path")
    parser.add_argument("--quantization", default="q4_k_m", 
                       choices=["q4_0", "q4_k_m", "q5_k_m", "q8_0", "f16"],
                       help="Quantization type")
    parser.add_argument("--skip-merge", action="store_true", help="Skip merge step if already done")
    parser.add_argument("--skip-gguf", action="store_true", help="Only merge, skip GGUF conversion")
    
    args = parser.parse_args()
    
    # Step 1: Merge LoRA weights
    if not args.skip_merge:
        merge_lora_weights(args.base_model, args.adapter_path, args.merged_output)
    else:
        print(f"Skipping merge, using existing: {args.merged_output}")
    
    # Step 2: Convert to GGUF
    if not args.skip_gguf:
        gguf_path = convert_to_gguf(args.merged_output, args.gguf_output, args.quantization)
        if gguf_path:
            print(f"\n{'='*60}")
            print(f"SUCCESS! GGUF model saved to: {gguf_path}")
            print(f"{'='*60}")
            print(f"\nYou can now use this model with the WebSocket server:")
            print(f'  MODEL_PATH = "{gguf_path}"')
    else:
        print(f"\nMerged model saved to: {args.merged_output}")
        print("Run GGUF conversion manually with llama.cpp")


if __name__ == "__main__":
    main()