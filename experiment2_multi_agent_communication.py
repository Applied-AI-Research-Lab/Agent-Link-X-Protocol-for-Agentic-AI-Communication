#!/usr/bin/env python3
"""
Multi-Agent Communication Efficiency
Evaluates encoding efficiency when multiple agents collaborate on tasks.
"""

import os
import base64
import gzip
import time
import json
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Configuration
API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
MODEL = "gpt-4o-mini"

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

# Agent Class

class Agent:
    """LLM-powered agent with encoding support."""
    
    def __init__(self, name: str, role: str, encoding: str, client: OpenAI, model: str):
        self.name = name
        self.role = role
        self.encoding = encoding
        self.client = client
        self.model = model
        self.message_history = []
        self.metrics = {
            "total_tokens": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_latency": 0,
            "message_count": 0
        }
    
    def encode_message(self, message: str) -> str:
        """Encode message according to agent's encoding type."""
        if self.encoding == "natural_language":
            return message
        elif self.encoding == "binary":
            return text_to_binary(message)
        elif self.encoding == "base64":
            return text_to_base64(message)
        elif self.encoding == "compressed_base64":
            return text_to_compressed_base64(message)
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")
    
    def send_message(self, content: str, to_agent: str, conversation_context: List[Dict] = None) -> Dict:
        """Send encoded message and track metrics."""
        
        # Encode the content
        encoded_content = self.encode_message(content)
        
        # Build conversation history
        messages = [
            {"role": "system", "content": f"You are {self.name}, a {self.role}. "
                                         f"Respond concisely and professionally."}
        ]
        
        # Add conversation context if provided
        if conversation_context:
            for msg in conversation_context[-3:]:  # Keep last 3 messages for context
                messages.append({"role": "user", "content": msg.get("encoded_content", "")})
                if "response_content" in msg:
                    messages.append({"role": "assistant", "content": msg["response_content"]})
        
        # Add current message
        messages.append({"role": "user", "content": encoded_content})
        
        # Call API
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            latency = time.time() - start_time
            
            # Update metrics
            self.metrics["total_tokens"] += response.usage.total_tokens
            self.metrics["total_input_tokens"] += response.usage.prompt_tokens
            self.metrics["total_output_tokens"] += response.usage.completion_tokens
            self.metrics["total_latency"] += latency
            self.metrics["message_count"] += 1
            
            response_content = response.choices[0].message.content
            
            message_record = {
                "from": self.name,
                "to": to_agent,
                "original_content": content,
                "encoded_content": encoded_content,
                "encoded_length": len(encoded_content),
                "response_content": response_content,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "latency": round(latency, 3),
                "success": True
            }
            
        except Exception as e:
            latency = time.time() - start_time
            self.metrics["total_latency"] += latency
            self.metrics["message_count"] += 1
            
            message_record = {
                "from": self.name,
                "to": to_agent,
                "original_content": content,
                "encoded_content": encoded_content,
                "encoded_length": len(encoded_content),
                "latency": round(latency, 3),
                "success": False,
                "error": str(e)
            }
        
        self.message_history.append(message_record)
        return message_record

# Multi-Agent Task Orchestration

class MultiAgentTask:
    """Orchestrate multi-agent collaborative task."""
    
    def __init__(self, encoding: str, client: OpenAI, model: str):
        self.encoding = encoding
        self.planner = Agent("Planner", "task planner and coordinator", encoding, client, model)
        self.worker = Agent("Worker", "implementation specialist", encoding, client, model)
        self.conversation_log = []
    
    def run_orbital_mechanics_task(self) -> Dict:
        """Execute collaborative orbital mechanics library development."""
        
        print(f"\n{'='*60}")
        print(f"Encoding: {self.encoding}")
        print(f"{'='*60}")
        
        task_description = """Develop 5 Python functions for orbital mechanics:
1. orbital_velocity(radius, central_mass) - velocity for circular orbit
2. escape_velocity(radius, central_mass) - minimum escape velocity
3. orbital_period(semi_major_axis, central_mass) - period of elliptical orbit
4. hohmann_delta_v(r1, r2, central_mass) - delta-v for Hohmann transfer
5. gravitational_force(m1, m2, distance) - force between two masses

Use SI units. Include the gravitational constant G = 6.67430e-11."""
        
        # Message 1: Planner assigns task
        print(f"[{self.planner.name}] Sending task specification...")
        msg1 = self.planner.send_message(
            f"{task_description}\n\nPlease implement function 1 first.",
            self.worker.name
        )
        self.conversation_log.append(msg1)
        if msg1["success"]:
            print(f"Tokens: {msg1['total_tokens']}, Latency: {msg1['latency']}s")
        else:
            print(f"Error: {msg1.get('error', 'Unknown')}")
            return self._generate_error_result("Message 1 failed")
        
        time.sleep(0.5)
        
        # Message 2: Worker implements function 1
        print(f"[{self.worker.name}] Implementing orbital_velocity...")
        msg2 = self.worker.send_message(
            "Here is orbital_velocity implementation:\n\n"
            "```python\n"
            "import math\n\n"
            "def orbital_velocity(radius, central_mass):\n"
            "    G = 6.67430e-11\n"
            "    return math.sqrt(G * central_mass / radius)\n"
            "```\n\n"
            "Function 1 complete. Ready for function 2.",
            self.planner.name,
            self.conversation_log
        )
        self.conversation_log.append(msg2)
        if msg2["success"]:
            print(f"Tokens: {msg2['total_tokens']}, Latency: {msg2['latency']}s")
        else:
            print(f"Error: {msg2.get('error', 'Unknown')}")
        
        time.sleep(0.5)
        
        # Message 3: Planner requests function 2
        print(f"[{self.planner.name}] Requesting function 2...")
        msg3 = self.planner.send_message(
            "Good. Now implement function 2: escape_velocity(radius, central_mass).",
            self.worker.name,
            self.conversation_log
        )
        self.conversation_log.append(msg3)
        if msg3["success"]:
            print(f"Tokens: {msg3['total_tokens']}, Latency: {msg3['latency']}s")
        else:
            print(f"Error: {msg3.get('error', 'Unknown')}")
        
        time.sleep(0.5)
        
        # Message 4: Worker implements function 2
        print(f"[{self.worker.name}] Implementing escape_velocity...")
        msg4 = self.worker.send_message(
            "Here is escape_velocity implementation:\n\n"
            "```python\n"
            "def escape_velocity(radius, central_mass):\n"
            "    G = 6.67430e-11\n"
            "    return math.sqrt(2 * G * central_mass / radius)\n"
            "```\n\n"
            "Function 2 complete. Ready for function 3.",
            self.planner.name,
            self.conversation_log
        )
        self.conversation_log.append(msg4)
        if msg4["success"]:
            print(f"Tokens: {msg4['total_tokens']}, Latency: {msg4['latency']}s")
        else:
            print(f"Error: {msg4.get('error', 'Unknown')}")
        
        time.sleep(0.5)
        
        # Message 5: Planner requests function 3
        print(f"[{self.planner.name}] Requesting function 3...")
        msg5 = self.planner.send_message(
            "Proceed with function 3: orbital_period(semi_major_axis, central_mass).",
            self.worker.name,
            self.conversation_log
        )
        self.conversation_log.append(msg5)
        if msg5["success"]:
            print(f"Tokens: {msg5['total_tokens']}, Latency: {msg5['latency']}s")
        else:
            print(f"Error: {msg5.get('error', 'Unknown')}")
        
        time.sleep(0.5)
        
        # Message 6: Worker implements function 3
        print(f"[{self.worker.name}] Implementing orbital_period...")
        msg6 = self.worker.send_message(
            "Here is orbital_period implementation:\n\n"
            "```python\n"
            "def orbital_period(semi_major_axis, central_mass):\n"
            "    G = 6.67430e-11\n"
            "    return 2 * math.pi * math.sqrt(semi_major_axis**3 / (G * central_mass))\n"
            "```\n\n"
            "Function 3 complete. Three functions done, two remaining.",
            self.planner.name,
            self.conversation_log
        )
        self.conversation_log.append(msg6)
        if msg6["success"]:
            print(f"Tokens: {msg6['total_tokens']}, Latency: {msg6['latency']}s")
        else:
            print(f"Error: {msg6.get('error', 'Unknown')}")
        
        time.sleep(0.5)
        
        # Message 7: Planner requests functions 4 and 5
        print(f"[{self.planner.name}] Requesting final functions...")
        msg7 = self.planner.send_message(
            "Implement the remaining two functions: hohmann_delta_v and gravitational_force.",
            self.worker.name,
            self.conversation_log
        )
        self.conversation_log.append(msg7)
        if msg7["success"]:
            print(f"Tokens: {msg7['total_tokens']}, Latency: {msg7['latency']}s")
        else:
            print(f"Error: {msg7.get('error', 'Unknown')}")
        
        time.sleep(0.5)
        
        # Message 8: Worker implements final functions
        print(f"[{self.worker.name}] Implementing final functions...")
        msg8 = self.worker.send_message(
            "Here are the final implementations:\n\n"
            "```python\n"
            "def hohmann_delta_v(r1, r2, central_mass):\n"
            "    G = 6.67430e-11\n"
            "    v1 = math.sqrt(G * central_mass / r1)\n"
            "    v_transfer_1 = math.sqrt(G * central_mass * (2/r1 - 2/(r1+r2)))\n"
            "    v2 = math.sqrt(G * central_mass / r2)\n"
            "    v_transfer_2 = math.sqrt(G * central_mass * (2/r2 - 2/(r1+r2)))\n"
            "    return abs(v_transfer_1 - v1) + abs(v2 - v_transfer_2)\n\n"
            "def gravitational_force(m1, m2, distance):\n"
            "    G = 6.67430e-11\n"
            "    return G * m1 * m2 / (distance ** 2)\n"
            "```\n\n"
            "All 5 functions complete. Library ready.",
            self.planner.name,
            self.conversation_log
        )
        self.conversation_log.append(msg8)
        if msg8["success"]:
            print(f"Tokens: {msg8['total_tokens']}, Latency: {msg8['latency']}s")
        else:
            print(f"Error: {msg8.get('error', 'Unknown')}")
        
        # Generate results
        return self._generate_result()
    
    def _generate_result(self) -> Dict:
        """Generate results dictionary."""
        total_metrics = {
            "encoding": self.encoding,
            "planner_metrics": self.planner.metrics,
            "worker_metrics": self.worker.metrics,
            "total_tokens": (self.planner.metrics["total_tokens"] + 
                           self.worker.metrics["total_tokens"]),
            "total_input_tokens": (self.planner.metrics["total_input_tokens"] + 
                                  self.worker.metrics["total_input_tokens"]),
            "total_output_tokens": (self.planner.metrics["total_output_tokens"] + 
                                   self.worker.metrics["total_output_tokens"]),
            "total_latency": round(self.planner.metrics["total_latency"] + 
                                 self.worker.metrics["total_latency"], 3),
            "total_messages": len(self.conversation_log),
            "conversation_log": self.conversation_log,
            "success": True
        }
        return total_metrics
    
    def _generate_error_result(self, error_msg: str) -> Dict:
        """Generate error result."""
        return {
            "encoding": self.encoding,
            "success": False,
            "error": error_msg,
            "total_tokens": 0,
            "total_latency": 0,
            "total_messages": len(self.conversation_log)
        }

# Main Analysis Function

def run_experiment_2(api_key: str, output_dir: str = "results"):
    """Run multi-agent communication analysis across all encoding types."""
    
    print("\n" + "="*70)
    print("Multi-Agent Communication Efficiency")
    print("="*70)
    
    client = OpenAI(api_key=api_key)
    
    encodings = ["natural_language", "binary", "base64", "compressed_base64"]
    all_results = []
    
    # Run task for each encoding
    for encoding in encodings:
        print(f"\n{'='*70}")
        print(f"Testing encoding: {encoding.upper()}")
        print(f"{'='*70}")
        
        task = MultiAgentTask(encoding, client, MODEL)
        result = task.run_orbital_mechanics_task()
        all_results.append(result)
        
        if result.get("success", False):
            print(f"\n Task completed successfully")
            print(f"Total tokens: {result['total_tokens']}")
            print(f"Total latency: {result['total_latency']:.2f}s")
            print(f"Total messages: {result['total_messages']}")
        else:
            print(f"\nTask failed: {result.get('error', 'Unknown error')}")
        
        # Delay between encoding tests
        time.sleep(2)
    
    # Generate Analysis
    
    print(f"\n{'='*70}")
    print("Generating Analysis")
    print(f"{'='*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate statistics
    summary = {
        "experiment": "Multi-Agent Communication Analysis",
        "timestamp": timestamp,
        "model": MODEL,
        "encoding_comparison": {}
    }
    
    baseline_tokens = None
    baseline_latency = None
    
    for result in all_results:
        if result.get("success", False):
            enc = result["encoding"]
            tokens = result["total_tokens"]
            latency = result["total_latency"]
            messages = result["total_messages"]
            
            summary["encoding_comparison"][enc] = {
                "total_tokens": tokens,
                "total_latency": round(latency, 3),
                "total_messages": messages,
                "avg_tokens_per_message": round(tokens / messages, 1) if messages > 0 else 0,
                "avg_latency_per_message": round(latency / messages, 3) if messages > 0 else 0
            }
            
            if enc == "natural_language":
                baseline_tokens = tokens
                baseline_latency = latency
    
    # Calculate efficiency gains
    if baseline_tokens and baseline_latency:
        print(f"\nBaseline (Natural Language):")
        print(f"Total tokens: {baseline_tokens}")
        print(f"Total latency: {baseline_latency:.3f}s")
        
        print(f"\nEfficiency Comparison:")
        for enc, data in summary["encoding_comparison"].items():
            if enc != "natural_language":
                tokens = data["total_tokens"]
                latency = data["total_latency"]
                
                token_change = ((tokens - baseline_tokens) / baseline_tokens) * 100
                latency_change = ((latency - baseline_latency) / baseline_latency) * 100
                
                data["token_change_pct"] = round(token_change, 1)
                data["latency_change_pct"] = round(latency_change, 1)
                
                print(f"\n{enc}:")
                print(f"Total tokens: {tokens} ({token_change:+.1f}%)")
                print(f"Total latency: {latency:.3f}s ({latency_change:+.1f}%)")
    
    # Save Results
    
    # Save detailed results
    detailed_file = os.path.join(output_dir, f"experiment2_detailed_{timestamp}.json")
    with open(detailed_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n Detailed results saved to: {detailed_file}")
    
    # Save summary
    summary_file = os.path.join(output_dir, f"experiment2_summary_{timestamp}.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f" Summary saved to: {summary_file}")
    
    # Generate LaTeX table
    latex_file = os.path.join(output_dir, f"experiment2_table_{timestamp}.tex")
    with open(latex_file, "w") as f:
        f.write("% Multi-Agent Communication Results\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Multi-Agent Communication Efficiency}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Encoding & Total Tokens & Token Change & Total Latency (s) & Latency Change \\\\\n")
        f.write("\\midrule\n")
        
        for enc, data in summary["encoding_comparison"].items():
            enc_display = enc.replace("_", " ").title()
            tokens = data["total_tokens"]
            latency = data["total_latency"]
            token_change = data.get("token_change_pct", 0)
            latency_change = data.get("latency_change_pct", 0)
            
            if enc == "natural_language":
                f.write(f"{enc_display} & {tokens} & baseline & {latency:.3f} & baseline \\\\\n")
            else:
                f.write(f"{enc_display} & {tokens} & {token_change:+.1f}\\% & {latency:.3f} & {latency_change:+.1f}\\% \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\label{tab:exp2_results}\n")
        f.write("\\end{table}\n")
    
    print(f" LaTeX table saved to: {latex_file}")
    
    print(f"\n{'='*70}")
    print("Analysis Complete")
    print(f"{'='*70}")
    
    return summary

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
    
    # Run analysis
    results = run_experiment_2(API_KEY)
    
    print("\n" + "="*70)
    print("Results saved to 'results' directory:")
    print("  - Detailed JSON results with conversation logs")
    print("  - Summary statistics")
    print("  - LaTeX table")
    print("="*70)
