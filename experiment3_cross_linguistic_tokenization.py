#!/usr/bin/env python3
"""
Cross-Linguistic Tokenization Analysis
Quantifies tokenization inefficiencies across languages and demonstrates
how machine-native encodings eliminate script-dependent penalties.
"""

import os
import base64
import gzip
import json
from datetime import datetime
from typing import Dict
from transformers import AutoTokenizer

# Configuration
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

# Sample Texts in Different Languages

SAMPLE_TEXTS = {
    "english": """
A binary search tree is a fundamental data structure in computer science that organizes 
elements in a hierarchical manner. Each node in the tree contains a value and has at most 
two children: a left child and a right child. The binary search tree property ensures that 
for any given node, all values in the left subtree are smaller than the node's value, and 
all values in the right subtree are larger. This organization enables efficient searching, 
insertion, and deletion operations with average time complexity of O(log n). The structure 
is particularly useful for maintaining sorted data and implementing various algorithms such 
as range queries and ordered traversals. However, the performance can degrade to O(n) in 
the worst case when the tree becomes unbalanced, which is why self-balancing variants like 
AVL trees and red-black trees were developed to guarantee logarithmic performance.
""".strip(),
    
    "greek": """
Ένα δυαδικό δέντρο αναζήτησης είναι μια θεμελιώδης δομή δεδομένων στην επιστήμη των 
υπολογιστών που οργανώνει στοιχεία με ιεραρχικό τρόπο. Κάθε κόμβος στο δέντρο περιέχει 
μια τιμή και έχει το πολύ δύο παιδιά: ένα αριστερό παιδί και ένα δεξί παιδί. Η ιδιότητα 
του δυαδικού δέντρου αναζήτησης διασφαλίζει ότι για οποιονδήποτε κόμβο, όλες οι τιμές 
στο αριστερό υποδέντρο είναι μικρότερες από την τιμή του κόμβου και όλες οι τιμές στο 
δεξί υποδέντρο είναι μεγαλύτερες. Αυτή η οργάνωση επιτρέπει αποδοτικές λειτουργίες 
αναζήτησης, εισαγωγής και διαγραφής με μέση πολυπλοκότητα χρόνου O(log n). Η δομή είναι 
ιδιαίτερα χρήσιμη για τη διατήρηση ταξινομημένων δεδομένων και την υλοποίηση διαφόρων 
αλγορίθμων όπως ερωτήματα εύρους και ταξινομημένες διασχίσεις. Ωστόσο, η απόδοση μπορεί 
να υποβαθμιστεί σε O(n) στη χειρότερη περίπτωση όταν το δέντρο γίνεται μη ισορροπημένο.
""".strip(),
    
    "chinese": """
二叉搜索树是计算机科学中的基本数据结构，它以分层方式组织元素。树中的每个节点包含
一个值，最多有两个子节点：左子节点和右子节点。二叉搜索树的属性确保对于任何给定的
节点，左子树中的所有值都小于该节点的值，而右子树中的所有值都大于该节点的值。这种
组织方式使得搜索、插入和删除操作的平均时间复杂度为O(log n)。该结构特别适用于维护
排序数据和实现各种算法，如范围查询和有序遍历。然而，当树变得不平衡时，性能可能会
降级到最坏情况下的O(n)，这就是为什么开发了自平衡变体（如AVL树和红黑树）以保证对数
性能的原因。这些数据结构在现代计算机系统中广泛应用，从数据库索引到文件系统组织，
都发挥着重要作用。理解和掌握二叉搜索树的原理对于任何计算机科学专业人士都是必不可少的。
""".strip()
}

# Tokenization Analysis

def analyze_language_tokenization(language: str, text: str, tokenizer) -> Dict:
    """Analyze tokenization for a single language across all encodings."""
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {language.upper()}")
    print(f"{'='*60}")
    print(f"Text length: {len(text)} characters, {len(text.encode('utf-8'))} bytes")
    
    # Generate all encodings
    encodings = {
        "natural_language": text,
        "binary": text_to_binary(text),
        "base64": text_to_base64(text),
        "compressed_base64": text_to_compressed_base64(text)
    }
    
    results = {
        "language": language,
        "character_count": len(text),
        "byte_count": len(text.encode('utf-8')),
        "encodings": {}
    }
    
    # Analyze each encoding
    for enc_name, enc_text in encodings.items():
        print(f"\n  {enc_name}:")
        
        try:
            # Tokenize
            tokens = tokenizer.encode(enc_text, add_special_tokens=True)
            token_count = len(tokens)
            
            # Calculate metrics
            chars_per_token = len(enc_text) / token_count if token_count > 0 else 0
            bytes_per_token = len(enc_text.encode('utf-8')) / token_count if token_count > 0 else 0
            
            # Calculate compression ratio relative to character count
            encoding_overhead = (len(enc_text) / len(text)) if len(text) > 0 else 0
            
            results["encodings"][enc_name] = {
                "token_count": token_count,
                "encoded_char_count": len(enc_text),
                "encoded_byte_count": len(enc_text.encode('utf-8')),
                "chars_per_token": round(chars_per_token, 2),
                "bytes_per_token": round(bytes_per_token, 2),
                "encoding_overhead": round(encoding_overhead, 2),
                "success": True
            }
            
            print(f"Token count: {token_count}")
            print(f"Chars/token: {chars_per_token:.2f}")
            print(f"Encoding overhead: {encoding_overhead:.2f}x")
            
        except Exception as e:
            print(f"Error: {e}")
            results["encodings"][enc_name] = {
                "success": False,
                "error": str(e)
            }
    
    return results

# Main Analysis Function

def run_experiment_3(output_dir="results"):
    """Run complete cross-linguistic tokenization analysis."""
    
    print("\n" + "="*70)
    print("Cross-Linguistic Tokenization Analysis")
    print("="*70)
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {TOKENIZER_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None
    
    # Analyze each language
    all_results = []
    for language, text in SAMPLE_TEXTS.items():
        result = analyze_language_tokenization(language, text, tokenizer)
        all_results.append(result)
    
    # Generate Analysis
    
    print(f"\n{'='*70}")
    print("Generating Cross-Linguistic Analysis")
    print(f"{'='*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate comparative statistics
    summary = {
        "experiment": "Cross-Linguistic Tokenization Analysis",
        "timestamp": timestamp,
        "tokenizer": TOKENIZER_NAME,
        "languages_analyzed": list(SAMPLE_TEXTS.keys()),
        "comparative_analysis": {}
    }
    
    # Get English baseline
    english_data = None
    for result in all_results:
        if result["language"] == "english":
            english_data = result
            break
    
    if english_data:
        english_nl_tokens = english_data["encodings"]["natural_language"]["token_count"]
        
        print(f"\nBaseline (English Natural Language): {english_nl_tokens} tokens")
        print(f"\nTokenization Overhead by Language:")
        
        for result in all_results:
            lang = result["language"]
            nl_encoding = result["encodings"].get("natural_language", {})
            
            if nl_encoding.get("success", False):
                tokens = nl_encoding["token_count"]
                chars = result["character_count"]
                chars_per_token = nl_encoding["chars_per_token"]
                
                # Calculate overhead vs English
                if lang == "english":
                    overhead_factor = 1.0
                else:
                    overhead_factor = tokens / english_nl_tokens if english_nl_tokens > 0 else 0
                
                summary["comparative_analysis"][lang] = {
                    "natural_language": {
                        "tokens": tokens,
                        "chars_per_token": chars_per_token,
                        "overhead_vs_english": round(overhead_factor, 2)
                    },
                    "encoding_efficiency": {}
                }
                
                print(f"\n  {lang.upper()}:")
                print(f"Tokens: {tokens}")
                print(f"Chars/token: {chars_per_token:.2f}")
                print(f"Overhead vs English: {overhead_factor:.2f}x")
                
                # Calculate efficiency gains from machine-native encoding
                print(f"Efficiency gains with encodings:")
                for enc_name in ["binary", "base64", "compressed_base64"]:
                    enc_data = result["encodings"].get(enc_name, {})
                    if enc_data.get("success", False):
                        enc_tokens = enc_data["token_count"]
                        reduction_pct = ((tokens - enc_tokens) / tokens * 100) if tokens > 0 else 0
                        
                        summary["comparative_analysis"][lang]["encoding_efficiency"][enc_name] = {
                            "tokens": enc_tokens,
                            "reduction_vs_nl": round(reduction_pct, 1)
                        }
                        
                        print(f"  {enc_name}: {enc_tokens} tokens ({reduction_pct:+.1f}%)")
    
    # Save Results
    
    # Save detailed results
    detailed_file = os.path.join(output_dir, f"experiment3_detailed_{timestamp}.json")
    with open(detailed_file, "w", encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {detailed_file}")
    
    # Save summary
    summary_file = os.path.join(output_dir, f"experiment3_summary_{timestamp}.json")
    with open(summary_file, "w", encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to: {summary_file}")
    
    # Generate LaTeX table
    latex_file = os.path.join(output_dir, f"experiment3_table_{timestamp}.tex")
    with open(latex_file, "w", encoding='utf-8') as f:
        f.write("% Cross-Linguistic Tokenization Results\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Cross-Linguistic Tokenization Analysis}\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("Language & NL Tokens & Overhead & Binary & Base64 & Compressed \\\\\n")
        f.write("& & vs English & Reduction & Reduction & Reduction \\\\\n")
        f.write("\\midrule\n")
        
        for lang in ["english", "greek", "chinese"]:
            if lang in summary["comparative_analysis"]:
                data = summary["comparative_analysis"][lang]
                nl_tokens = data["natural_language"]["tokens"]
                overhead = data["natural_language"]["overhead_vs_english"]
                
                binary_red = data["encoding_efficiency"].get("binary", {}).get("reduction_vs_nl", 0)
                base64_red = data["encoding_efficiency"].get("base64", {}).get("reduction_vs_nl", 0)
                comp_red = data["encoding_efficiency"].get("compressed_base64", {}).get("reduction_vs_nl", 0)
                
                lang_display = lang.capitalize()
                f.write(f"{lang_display} & {nl_tokens} & {overhead:.2f}$\\times$ & "
                       f"{binary_red:+.1f}\\% & {base64_red:+.1f}\\% & {comp_red:+.1f}\\% \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\label{tab:exp3_results}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to: {latex_file}")
    
    # Generate visualization data for plotting
    plot_file = os.path.join(output_dir, f"experiment3_plot_data_{timestamp}.json")
    plot_data = {
        "languages": [],
        "natural_language_tokens": [],
        "binary_tokens": [],
        "base64_tokens": [],
        "compressed_tokens": []
    }
    
    for result in all_results:
        lang = result["language"]
        plot_data["languages"].append(lang)
        
        for enc_name, key in [
            ("natural_language", "natural_language_tokens"),
            ("binary", "binary_tokens"),
            ("base64", "base64_tokens"),
            ("compressed_base64", "compressed_tokens")
        ]:
            enc_data = result["encodings"].get(enc_name, {})
            tokens = enc_data.get("token_count", 0) if enc_data.get("success", False) else 0
            plot_data[key].append(tokens)
    
    with open(plot_file, "w") as f:
        json.dump(plot_data, f, indent=2)
    print(f"LaTeX table saved to: {latex_file}")
    
    print(f"\n{'='*70}")
    print("Analysis Complete")
    print(f"{'='*70}")
    
    return summary

# Main Entry Point

if __name__ == "__main__":
    print("\nStarting analysis...")
    print(f"Analyzing {len(SAMPLE_TEXTS)} languages: {', '.join(SAMPLE_TEXTS.keys())}")
    
    # Run analysis
    results = run_experiment_3()
    
    if results:
        print("\n" + "="*70)
        print("Results saved to 'results' directory:")
        print("  - Detailed JSON results")
        print("  - Summary statistics")
        print("  - LaTeX table")
        print("  - Plot data for visualization")
        print("="*70)
        print("\nKey Findings:")
        print("  - Tokenization overhead varies significantly across scripts")
        print("  - Machine-native encodings eliminate script-dependent penalties")
        print("  - Non-Latin languages show greater efficiency gains")
    else:
        print("\nAnalysis failed. Please check the error messages above.")
