import os
import json
from google import genai
from google.genai import types
import time

# Configuration
MASTER_FILE = './master_sample.json'
OUTPUT_FILE = './generated_prompts.txt'
THOUGHT_SIGNATURES_FILE = './thought_signatures.json'

EXCLUDED_MODELS = {
    'gemini', 
    'gpt', 
    'stable-diffusion-3.5-medium', 
    'stable-diffusion-xl-1.0-inpainting-0.1'
}

def format_item(item):
    manipulation = item.get('manipulation', 'N/A')
    if item['class'] == 'Real':
        manipulation = 'N/A'
    
    return f"""[Example ID: {item['id']}]
- Class: {item['class']}
- Modality: {item['modality']}
- Manipulation: {manipulation}
- Model: {item['model']}
- Evidence: {item['evidence']}
"""

def generate_content(client, system_prompt, items):
    input_text = "\n".join([format_item(i) for i in items])
    
    full_prompt = f"""{system_prompt}

Here are the examples:
{input_text}
"""
    try:
        response = client.models.generate_content(
            model='gemini-3-pro-preview',
            contents=full_prompt,
            # config=types.GenerateContentConfig(
            #     temperature=0.1,
            #     top_p=1,
            #     top_k=32,
            #     max_output_tokens=1024,
            #     thinking_config=types.ThinkingConfig(thinking_budget=512, include_thoughts=True)
            # )
        )
        
        # Properly extract text from response, handling thought_signature parts
        text_parts = []
        thought_signatures = []
        
        if response.candidates and len(response.candidates) > 0:
            for part in response.candidates[0].content.parts:
                # Extract text
                if hasattr(part, 'text') and part.text:
                    text_parts.append(part.text)
                # Extract thought_signature (may be a string or object)
                if hasattr(part, 'thought_signature'):
                    sig = part.thought_signature
                    if sig:  # Only add non-empty signatures
                        # Convert to string if it's not already
                        thought_signatures.append(str(sig) if not isinstance(sig, str) else sig)
        
        # Fallback to response.text if parts extraction didn't work
        result_text = ''.join(text_parts) if text_parts else response.text
        
        # Return both text and thought_signature
        return result_text, thought_signatures
    except Exception as e:
        print(f"Error generating content: {e}")
        return "", []

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set.")
        return

    client = genai.Client(api_key=api_key)

    with open(MASTER_FILE, 'r') as f:
        data = json.load(f)

    prompts = {}
    thought_signatures = {}  # Store thought signatures by key

    # Unified system prompt for all three passes
    unified_system_prompt = (
        "You are a medical forensics expert. Analyze these medical forensics examples (Real and Fake) in great detail. "
        "Provide a comprehensive, detailed summary of the universal principles, visual artifacts (e.g., smoothing, boundaries, texture inconsistencies), "
        "and diagnostic features for detecting deepfakes. "
        "Cover both lesion removal and implant manipulations across all generator architectures with specific examples. "
        "Also highlight features of authentic images (e.g. correct anatomy, mass effect, proper density gradients). "
        "Be thorough and exhaustive in your analysis. Include specific artifact patterns, anatomical violations, and imaging physics principles. "
        "The output should be a comprehensive, detailed paragraph suitable for in-context learning. Make it as detailed and complete as possible. Output a very very long  10K word guidance for deepfake detection."
    )

    # --- Pass 1: In Domain ---
    print("Generating In-Domain Prompt...")
    text, sigs = generate_content(client, unified_system_prompt, data)
    prompts['In Domain'] = text
    thought_signatures['In Domain'] = sigs

    # --- Pass 2: Cross Model ---
    print("Generating Cross-Model Prompt...")
    # Filter: Remove excluded models, keep Real and non-excluded Fake
    subset_2 = [x for x in data if x['model'] not in EXCLUDED_MODELS]
    text, sigs = generate_content(client, unified_system_prompt, subset_2)
    prompts['Cross Model'] = text
    thought_signatures['Cross Model'] = sigs

    # --- Pass 3: Cross Forgery ---
    print("Generating Cross-Forgery Prompt...")
    # Filter: Remove '*-edit' (Implant). Keep '*-remove' and Real.
    subset_3 = [x for x in data if 'edit' not in x.get('type', '')]
    text, sigs = generate_content(client, unified_system_prompt, subset_3)
    prompts['Cross Forgery'] = text
    thought_signatures['Cross Forgery'] = sigs


    # Save Outputs
    with open(OUTPUT_FILE, 'w') as f:
        for key, val in prompts.items():
            f.write(f"=== {key} ===\n\n{val}\n\n")
    
    # Save thought signatures
    with open(THOUGHT_SIGNATURES_FILE, 'w') as f:
        json.dump(thought_signatures, f, indent=2, ensure_ascii=False)
        
    print(f"Prompts saved to {OUTPUT_FILE}")
    print(f"Thought signatures saved to {THOUGHT_SIGNATURES_FILE}")

if __name__ == "__main__":
    main()

