#!/usr/bin/env python3
"""
Message Encoding Cost Analysis
Compares computational costs between natural language and machine-native encodings
for single-turn interactions with language models.
"""

import os
import base64
import gzip
import time
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer

load_dotenv()

# Configuration
API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
MODEL = "gpt-4o-mini"
TOKENIZER_NAME = "gpt2"

# Encoding Utilities

def text_to_binary(text):
    """Convert text to space-separated binary string."""
    binary = ' '.join(format(ord(c), '08b') for c in text)
    return binary

def text_to_base64(text):
    """Convert text to base64 encoding."""
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return encoded

def text_to_compressed_base64(text):
    """Convert text to gzip-compressed base64."""
    compressed = gzip.compress(text.encode("utf-8"))
    encoded = base64.b64encode(compressed).decode("ascii")
    return encoded

# Tokenization Analysis

def analyze_encoding_local(text, encoding_name, tokenizer):
    """Analyze token count and metrics using local tokenizer."""
    try:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        token_count = len(tokens)
    except Exception as e:
        print(f"Warning: Tokenization failed for {encoding_name}: {e}")
        token_count = 0
    
    byte_count = len(text.encode('utf-8'))
    chars_per_token = len(text) / token_count if token_count > 0 else 0
    bytes_per_token = byte_count / token_count if token_count > 0 else 0
    
    return {
        "encoding": encoding_name,
        "token_count_local": token_count,
        "character_count": len(text),
        "byte_count": byte_count,
        "chars_per_token": round(chars_per_token, 2),
        "bytes_per_token": round(bytes_per_token, 2)
    }

# LLM API Measurement

def measure_llm_response(client, message, encoding_type, model):
    """Measure LLM response time and token usage."""
    
    # Construct prompt based on encoding type
    if encoding_type == "natural_language":
        prompt = f"Process this message and provide a one-sentence summary:\n\n{message}"
    else:
        prompt = f"This is a {encoding_type} encoded message. Decode and process it, then provide a one-sentence summary:\n\n{message}"
    
    # Measure API call
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        latency = time.time() - start_time
        
        # Extract metrics
        return {
            "encoding_type": encoding_type,
            "latency_seconds": round(latency, 3),
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "response": response.choices[0].message.content,
            "success": True
        }
    except Exception as e:
        latency = time.time() - start_time
        return {
            "encoding_type": encoding_type,
            "latency_seconds": round(latency, 3),
            "success": False,
            "error": str(e)
        }

# Main Analysis Function

def run_experiment_1(input_texts, api_key, output_dir="results"):
    """Run complete encoding cost comparison analysis."""
    
    print("\n" + "="*70)
    print("Message Encoding Cost Analysis")
    print("="*70)
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Load tokenizer for local analysis
    print(f"\nLoading tokenizer: {TOKENIZER_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        print("Tokenizer loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")
        print("Continuing without local tokenization analysis...")
        tokenizer = None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    all_results = []
    
    # Process each input text
    for idx, original_text in enumerate(input_texts, 1):
        print(f"\n{'='*70}")
        print(f"Processing Message {idx}/{len(input_texts)}")
        print(f"Original length: {len(original_text)} characters")
        print(f"{'='*70}")
        
        message_results = {
            "message_id": idx,
            "original_text": original_text[:100] + "..." if len(original_text) > 100 else original_text,
            "original_length": len(original_text),
            "encodings": {}
        }
        
        # Generate all encodings
        encodings = {
            "natural_language": original_text,
            "binary": text_to_binary(original_text),
            "base64": text_to_base64(original_text),
            "compressed_base64": text_to_compressed_base64(original_text)
        }
        
        # Analyze each encoding
        for enc_type, encoded_message in encodings.items():
            print(f"\n--- Testing {enc_type} ---")
            
            # Local tokenization analysis
            local_analysis = None
            if tokenizer:
                local_analysis = analyze_encoding_local(encoded_message, enc_type, tokenizer)
                print(f"Local token count: {local_analysis['token_count_local']}")
                print(f"Encoded length: {local_analysis['character_count']} chars, {local_analysis['byte_count']} bytes")
            
            # API measurement
            print(f"Calling API with {MODEL}...")
            api_result = measure_llm_response(client, encoded_message, enc_type, MODEL)
            
            if api_result["success"]:
                print(f"✓ Success")
                print(f"  Latency: {api_result['latency_seconds']:.3f}s")
                print(f"  Input tokens: {api_result['input_tokens']}")
                print(f"  Total tokens: {api_result['total_tokens']}")
            else:
                print(f"✗ Error: {api_result['error']}")
            
            # Combine results
            message_results["encodings"][enc_type] = {
                "local_analysis": local_analysis,
                "api_measurement": api_result
            }
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
        
        all_results.append(message_results)
    
    # Generate Analysis and Reports
    
    print(f"\n{'='*70}")
    print("Generating Analysis")
    print(f"{'='*70}")
    
    # Calculate statistics
    summary = {
        "experiment": "Message Encoding Cost Analysis",
        "timestamp": timestamp,
        "model": MODEL,
        "num_messages": len(input_texts),
        "encoding_comparison": {}
    }
    
    # Aggregate metrics by encoding type
    for enc_type in ["natural_language", "binary", "base64", "compressed_base64"]:
        latencies = []
        input_tokens = []
        total_tokens = []
        success_count = 0
        
        for msg_result in all_results:
            api_result = msg_result["encodings"][enc_type]["api_measurement"]
            if api_result["success"]:
                latencies.append(api_result["latency_seconds"])
                input_tokens.append(api_result["input_tokens"])
                total_tokens.append(api_result["total_tokens"])
                success_count += 1
        
        if success_count > 0:
            summary["encoding_comparison"][enc_type] = {
                "avg_latency": round(sum(latencies) / len(latencies), 3),
                "avg_input_tokens": round(sum(input_tokens) / len(input_tokens), 1),
                "avg_total_tokens": round(sum(total_tokens) / len(total_tokens), 1),
                "success_rate": f"{success_count}/{len(input_texts)}"
            }
    
    # Calculate efficiency gains
    if "natural_language" in summary["encoding_comparison"]:
        baseline_tokens = summary["encoding_comparison"]["natural_language"]["avg_total_tokens"]
        baseline_latency = summary["encoding_comparison"]["natural_language"]["avg_latency"]
        
        print(f"\nBaseline (Natural Language):")
        print(f"  Avg tokens: {baseline_tokens:.1f}")
        print(f"  Avg latency: {baseline_latency:.3f}s")
        
        print(f"\nEfficiency Comparison:")
        for enc_type in ["binary", "base64", "compressed_base64"]:
            if enc_type in summary["encoding_comparison"]:
                tokens = summary["encoding_comparison"][enc_type]["avg_total_tokens"]
                latency = summary["encoding_comparison"][enc_type]["avg_latency"]
                
                token_change = ((tokens - baseline_tokens) / baseline_tokens) * 100
                latency_change = ((latency - baseline_latency) / baseline_latency) * 100
                
                summary["encoding_comparison"][enc_type]["token_change_pct"] = round(token_change, 1)
                summary["encoding_comparison"][enc_type]["latency_change_pct"] = round(latency_change, 1)
                
                print(f"\n{enc_type}:")
                print(f"  Avg tokens: {tokens:.1f} ({token_change:+.1f}%)")
                print(f"  Avg latency: {latency:.3f}s ({latency_change:+.1f}%)")
    
    # Save Results
    
    # Save detailed results
    detailed_file = os.path.join(output_dir, f"experiment1_detailed_{timestamp}.json")
    with open(detailed_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Detailed results saved to: {detailed_file}")
    
    # Save summary
    summary_file = os.path.join(output_dir, f"experiment1_summary_{timestamp}.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to: {summary_file}")
    
    # Generate LaTeX table
    latex_file = os.path.join(output_dir, f"experiment1_table_{timestamp}.tex")
    with open(latex_file, "w") as f:
        f.write("% Encoding Efficiency Comparison Results\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Encoding Efficiency Comparison}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Encoding & Avg Tokens & Token Change & Avg Latency (s) & Latency Change \\\\\n")
        f.write("\\midrule\n")
        
        for enc_type, data in summary["encoding_comparison"].items():
            enc_display = enc_type.replace("_", " ").title()
            tokens = data["avg_total_tokens"]
            latency = data["avg_latency"]
            token_change = data.get("token_change_pct", 0)
            latency_change = data.get("latency_change_pct", 0)
            
            if enc_type == "natural_language":
                f.write(f"{enc_display} & {tokens:.1f} & baseline & {latency:.3f} & baseline \\\\\n")
            else:
                f.write(f"{enc_display} & {tokens:.1f} & {token_change:+.1f}\\% & {latency:.3f} & {latency_change:+.1f}\\% \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\label{tab:exp1_results}\n")
        f.write("\\end{table}\n")
    
    print(f"✓ LaTeX table saved to: {latex_file}")
    
    print(f"\n{'='*70}")
    print("Analysis Complete")
    print(f"{'='*70}")
    
    return summary

# Sample Input Texts

SAMPLE_TEXTS = [
    """Develop a Python function that calculates the Fibonacci sequence up to n terms. 
    The function should handle edge cases and be optimized for performance. Include 
    proper documentation and type hints.""",
    
    """Explain the concept of transformer attention mechanisms in neural networks. 
    Describe how self-attention works and why it has been so effective for natural 
    language processing tasks.""",
    
    """Write a SQL query that joins three tables: users, orders, and products. 
    The query should return the total sales amount per user for the last 30 days, 
    sorted by sales amount in descending order.""",
    
    """Design a REST API endpoint for a user authentication system. Include the 
    request/response format, status codes, error handling, and security considerations 
    such as token-based authentication.""",
    
    """Implement a binary search tree in Python with methods for insertion, deletion, 
    and in-order traversal. Ensure the implementation maintains BST properties and 
    handles edge cases appropriately."""
]

# Main Entry Point

if __name__ == "__main__":
    # Check API key
    if API_KEY == "your-api-key-here":
        print("ERROR: Please set your OpenAI API key!")
        print("Either:")
        print("  1. Set OPENAI_API_KEY environment variable")
        print("  2. Edit the API_KEY variable in this script")
        exit(1)
    
    print("\nStarting analysis...")
    print(f"Using model: {MODEL}")
    print(f"Number of test messages: {len(SAMPLE_TEXTS)}")
    
    # Run analysis
    results = run_experiment_1(SAMPLE_TEXTS, API_KEY)
    
    print("\n" + "="*70)
    print("Results saved to 'results' directory:")
    print("  - Detailed JSON results")
    print("  - Summary statistics")
    print("  - LaTeX table")
    print("="*70)
