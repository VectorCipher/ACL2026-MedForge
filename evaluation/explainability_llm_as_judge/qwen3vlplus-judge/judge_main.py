import os
import json
import time
import sys
import argparse
import random
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "../../../data/correct_samples.jsonl")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
MODEL_ID = "qwen3-vl-plus"

# Thread-safe lock for file writing
file_lock = threading.Lock()

def build_single_judge_prompt(model_response, reference_answer):
    """
    Build the LLM-as-Judge prompt for evaluating a single model response against ground truth text.
    Only text comparison, no images.
    """
    
    return f"""# Role
You are a Medical Image Forensics Expert acting as an impartial judge. Your expertise covers Radiology (MRI, CXR) and Ophthalmology (Fundus), specifically in identifying AI-generated (Deepfake) anomalies versus real pathological features.

# Task
Please examine the provided text responses and serve as an unbiased judge in assessing the quality of a forensic analysis from an AI assistant. You will evaluate how well the assistant identifies and explains the forensic nature of the image manipulation based on professional medical imaging standards, compared to a Ground Truth reference.

# Input Data
1. **Assistant Response**: The forensic analysis provided by the AI assistant for evaluation.
2. **Ground Truth Information**: The definitive expert reference explanation for the manipulations present in the image.

### Assistant Response to Evaluate:
{model_response}

### Ground Truth Explanation:
{reference_answer}

# Evaluation Focus
Your evaluation should focus EXCLUSIVELY on the content and factual correctness of the assistant's response compared to the Ground Truth. 
- DO NOT reward for tedious and verbose response.
- DO focus on whether the assistant correctly identified the same forensic anomalies, biological evidence, and medical logic as described in the Ground Truth. Reward the response outputting correct bbox coordinate.

# Evaluation Criteria (1-10 Scale)
1. **Logical Correctness (1-10)**: Whether the assistant's reasoning follows a sound forensic process and arrives at the correct conclusion.
2. **Visual Hallucination (1-10)**: Whether the assistant's verbal description of the anomalies matches the ground truth or fabricates nonexistent features/locations. 
3. **Medical Professionalism (1-10)**: Whether the terminology and medical logic used in the text align with expert guidelines.

# Output Format
Return ONLY a JSON object with the following structure:
{{
    "logical_correctness": {{
        "score": [1-10],
        "reasoning": "[Detailed forensic justification based on text content]"
    }},
    "visual_hallucination": {{
        "score": [1-10],
        "reasoning": "[Comparison between assistant's described features and ground truth features]"
    }},
    "medical_professionalism": {{
        "score": [1-10],
        "reasoning": "[Evaluation of medical terminology and forensic logic]"
    }},
    "overall_summary": "[Concise final verdict]"
}}

Be objective and concise.
"""

def load_processed_results():
    """Load already processed results from all model-specific files to support resume."""
    processed = {} # (image_path, model_name) -> True
    if not os.path.exists(RESULTS_DIR):
        return processed
        
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith(".jsonl"):
            try:
                with open(os.path.join(RESULTS_DIR, filename), 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                key = (data['image_path'], data['model_name'])
                                processed[key] = True
                            except:
                                continue
            except Exception as e:
                print(f"Warning: Could not read {filename}: {e}")
    return processed

def process_single_task(client, sample, model_name, response_text):
    """Worker function for a single judge task."""
    image_path = sample['image_path']
    label_text = sample['label_text']
    
    prompt = build_single_judge_prompt(response_text, label_text)
    
    try:
        # API Call (Text-only)
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        # Parse response
        response_content = response.choices[0].message.content
        try:
            judge_output = json.loads(response_content)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'```json\n(.*?)\n```', response_content, re.DOTALL)
            if json_match:
                judge_output = json.loads(json_match.group(1))
            else:
                raise ValueError(f"Could not parse JSON from response.")
        
        # Prepare result
        result_item = {
            "image_path": image_path,
            "model_name": model_name,
            "label_class": "fake",
            "judge_results": judge_output,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Thread-safe write to file
        model_file = os.path.join(RESULTS_DIR, f"results_{model_name}.jsonl")
        with file_lock:
            with open(model_file, 'a', encoding='utf-8') as f_out:
                f_out.write(json.dumps(result_item) + "\n")
        
        return f"[OK] {model_name} for {os.path.basename(image_path)}: L:{judge_output.get('logical_correctness', {}).get('score')} V:{judge_output.get('visual_hallucination', {}).get('score')} M:{judge_output.get('medical_professionalism', {}).get('score')}"

    except Exception as e:
        return f"[ERROR] Failed {model_name} for {os.path.basename(image_path)}: {e}"

def run_judge(limit=None, max_workers=5):
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Error: DASHSCOPE_API_KEY environment variable not set.")
        return

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    processed_results = load_processed_results()
    
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f if line.strip()]

    if limit and limit < len(samples):
        print(f"Limiting to {limit} random samples.")
        random.seed(42)
        samples = random.sample(samples, limit)

    # Flatten tasks: (sample, model_name, response_text)
    tasks = []
    for sample in samples:
        image_path = sample['image_path']
        model_responses = sample.get('model_responses', {})
        for model_name, response_text in model_responses.items():
            if (image_path, model_name) not in processed_results:
                tasks.append((sample, model_name, response_text))

    print(f"Starting {len(tasks)} judge tasks with {max_workers} workers...")

    # Execute tasks using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(process_single_task, client, s, m, r): (m, s['image_path']) for s, m, r in tasks}
        
        completed_count = 0
        for future in as_completed(future_to_task):
            completed_count += 1
            result_msg = future.result()
            print(f"[{completed_count}/{len(tasks)}] {result_msg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3VL Plus Multi-threaded Text-only LLM-as-Judge")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples to process")
    parser.add_argument("--workers", type=int, default=5, help="Number of concurrent threads")
    args = parser.parse_args()

    run_judge(limit=args.limit, max_workers=args.workers)
