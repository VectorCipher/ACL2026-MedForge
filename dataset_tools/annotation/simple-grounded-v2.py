
import base64
import requests
import os
import json
from pathlib import Path
from PIL import Image
from io import BytesIO
import numpy as np

# Gemini API related imports
from google import genai
from google.genai import types

# Qwen API related imports
from openai import OpenAI


# --- Helper Functions ---
def encode_image(image_path):
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Define mapping from dataset_type to modality name
modality_mapping = {
    "brain-mri-edit": "Brain MRI",
    "brain-mri-remove": "Brain MRI",
    "chest-xray-edit": "Chest X-Ray",
    "chest-xray-remove": "Chest X-Ray",
    "fundus-edit": "Fundus Photography",
    "fundus-remove": "Fundus Photography",
}

def load_full_guidelines_content(guideline_file_path):
    """
    Read complete medical Deepfake detection guidelines from markdown file.
    """
    with open(guideline_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def get_filtered_guidelines(full_guidelines_content, dataset_type):
    """
    Extract General Principles and corresponding Modality-Specific Criteria from full guidelines content based on dataset_type.
    """
    modality_name = modality_mapping.get(dataset_type)
    
    general_principles_start = full_guidelines_content.find("## General Principles (Universal Criteria)")
    modality_specific_start = full_guidelines_content.find("## Modality-Specific Criteria")

    filtered_section = ""

    # Extract General Principles
    if general_principles_start != -1:
        general_principles_end = full_guidelines_content.find("---", general_principles_start)
        if general_principles_end != -1:
            filtered_section += full_guidelines_content[general_principles_start:general_principles_end].strip()
        else:
            filtered_section += full_guidelines_content[general_principles_start:].strip()
    
    # Add Modality-Specific Criteria if a mapping exists
    if modality_name and modality_specific_start != -1:
        target_modality_heading = ""
        if modality_name == "Brain MRI":
            target_modality_heading = "### Brain MRI (Magnetic Resonance Imaging)"
        elif modality_name == "Fundus Photography":
            target_modality_heading = "### Fundus Photography (Retinal Imaging)"
        elif modality_name == "Chest X-Ray":
            target_modality_heading = "### Chest X-Ray (CXR)"
        
        if target_modality_heading:
            modality_start = full_guidelines_content.find(target_modality_heading, modality_specific_start)
            if modality_start != -1:
                next_modality_or_end = full_guidelines_content.find("---", modality_start)
                modality_section_content = ""
                if next_modality_or_end == -1:
                    modality_section_content = full_guidelines_content[modality_start:].strip()
                else:
                    modality_section_content = full_guidelines_content[modality_start:next_modality_or_end].strip()
                
                if filtered_section: # Add a separator if General Principles already exist
                    filtered_section += "\n\n"
                filtered_section += modality_section_content
                
    return filtered_section.strip()

def extract_bbox_from_mask(mask_path, image_size=1024):
    """
    Extract bounding box (bbox) from black and white mask image.
    
    Args:
        mask_path (str): Path to mask image
        image_size (int): Image size, default 1024
        
    Returns:
        tuple: (bbox_abs, bbox_normalized)
            - bbox_abs: Absolute coordinates [xmin, ymin, xmax, ymax]
            - bbox_normalized: Normalized coordinates to 0-1000 [ymin, xmin, ymax, xmax]
    """
    mask = Image.open(mask_path).convert('L')
    mask_array = np.array(mask)
    
    # Find positions of all non-zero pixels
    coords = np.where(mask_array > 0)
    
    if len(coords[0]) == 0:
        # If no non-zero pixels found, return entire image
        return [0, 0, image_size, image_size], [0, 0, 1000, 1000]
    
    # Calculate bounding box (absolute coordinates)
    ymin = int(np.min(coords[0]))
    ymax = int(np.max(coords[0]))
    xmin = int(np.min(coords[1]))
    xmax = int(np.max(coords[1]))
    
    bbox_abs = [xmin, ymin, xmax, ymax]
    
    # Normalize to 0-1000 (based on image_size)
    height, width = mask_array.shape
    print(f"Image height: {height}, Image width: {width}")
    bbox_normalized = [
        int(ymin / height * 1000),  # ymin
        int(xmin / width * 1000),   # xmin
        int(ymax / height * 1000),  # ymax
        int(xmax / width * 1000)    # xmax
    ]
    
    return bbox_abs, bbox_normalized

def crop_image_with_bbox(image_path, bbox, output_path):
    """
    Crop image using bbox and save.
    
    Args:
        image_path (str): Original image path
        bbox (list): Bounding box [xmin, ymin, xmax, ymax]
        output_path (str): Path to save cropped image
    """
    image = Image.open(image_path)
    xmin, ymin, xmax, ymax = bbox
    
    # Ensure coordinates are within image bounds
    width, height = image.size
    xmin = max(0, min(xmin, width))
    ymin = max(0, min(ymin, height))
    xmax = max(xmin, min(xmax, width))
    ymax = max(ymin, min(ymax, height))
    
    # Crop image
    cropped = image.crop((xmin, ymin, xmax, ymax))
    
    # Save cropped result
    cropped.save(output_path)
    print(f"Cropped image saved to: {output_path}")
    
    return cropped

def build_deepfake_analysis_prompt(bbox_normalized, guidelines):
    """
    Build prompt for analyzing deepfake images.
    
    Args:
        bbox_normalized (list): Normalized bbox values to 0-1000 [ymin, xmin, ymax, xmax].
        guidelines (str): Deepfake detection guideline text.
        
    Returns:
        str: Complete prompt text.
    """
    full_prompt = f"""This is a medical deepfake image. The bounding box [{bbox_normalized[0]}, {bbox_normalized[1]}, {bbox_normalized[2]}, {bbox_normalized[3]}] indicates the location of the deepfake region.

Your task is to analyze why this image is a deepfake by systematically applying the Medical Deepfake Detection Guidelines below. 

The output should adhere to the following format:
This image contains deepfake regions. Location: <seg>\n <box class=\"deepfake\" x1=\"{bbox_normalized[1]}\" y1=\"{bbox_normalized[0]}\" x2=\"{bbox_normalized[3]}\" y2=\"{bbox_normalized[2]}\" />\n</seg> 
Description: (Briefly describe the image, its modality, features, and deepfake region.)
Key Explanation:
   The anatomical errors present in this image are as follows:
   - (Identify anomalies according to the guidelines if any.)
   - (Describe the specific anomalies observed in the deepfake region.)
   - (Identify additional anomalies in the image that are not mentioned in the guidelines if any.)
Conclusion:
   (The conclusion that this image is a deepfake.)
     
The answer should be concise and to the point. Please provide the most relevant clues that indicate this is a fake image, avoiding vague or uninformative responses. Do not repeat titles or headers from the guidelines.
Medical Deepfake Detection Clues:
{guidelines}
"""
    return full_prompt

# --- Gemini API Request Functions ---
def analyze_deepfake_image_gemini(image_path, bbox_normalized, api_key, guidelines, enable_thinking=True, thinking_budget=512):
    """
    Analyze Deepfake images using Gemini API.

    Args:
        image_path (str): Path to original image file.
        bbox_normalized (list): Normalized bbox values to 0-1000 [ymin, xmin, ymax, xmax].
        api_key (str): Your Gemini API key.
        guidelines (str): Deepfake detection guideline text.
        enable_thinking (bool): Whether to enable thinking process, default True.
        thinking_budget (int): Token budget for thinking process, default 512.

    Returns:
        tuple: (Complete response object, serializable response dictionary)
    """

    client = genai.Client(api_key=api_key)

    image_obj = Image.open(image_path)

    full_prompt = build_deepfake_analysis_prompt(bbox_normalized, guidelines)

    response = client.models.generate_content(
        model="gemini-2.5-pro", # Suitable for image and text input
        contents=[
            image_obj,
            full_prompt
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,
            top_p=1,
            top_k=32,
            max_output_tokens=1024,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget,
                                                 include_thoughts=enable_thinking) 
        )
    )
    
    # Extract reasoning content and answer content (for display)
    reasoning_content = ""
    answer_content = ""
    
    if response.candidates and len(response.candidates) > 0:
        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                reasoning_content += part.text
            else:
                answer_content += part.text
    
    # Convert entire response object to string
    response_str = str(response)
    
    # Return dictionary containing stringified response
    response_dict = {
        "full_response_str": response_str,
        "extracted_reasoning": reasoning_content,
        "extracted_answer": answer_content,
        "request_prompt": full_prompt
    }
    
    return response, response_dict

# --- Qwen API Request Functions ---
def analyze_deepfake_image_qwen(image_path, bbox_normalized, api_key, base_url, guidelines, enable_thinking=True, thinking_budget=512):
    """
    Analyze Deepfake images using Qwen API.

    Args:
        image_path (str): Path to original image file.
        bbox_normalized (list): Normalized bbox values to 0-1000 [ymin, xmin, ymax, xmax].
        api_key (str): Your Qwen API key (DASHSCOPE_API_KEY).
        base_url (str): Base URL for Qwen API.
        guidelines (str): Deepfake detection guideline text.
        enable_thinking (bool): Whether to enable thinking process, default True.
        thinking_budget (int): Token budget for thinking process, default 512.

    Returns:
        dict: Complete API response information, including all chunks and usage information.
    """
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    base64_image = encode_image(image_path)

    full_prompt = build_deepfake_analysis_prompt(bbox_normalized, guidelines)

    response = client.chat.completions.create(
        model="qwen3-vl-flash", # Assume using Qwen-VL model that supports image input
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                    {"type": "text", "text": full_prompt},
                ],
            },
        ],
        max_tokens=1024,
        temperature = 0.1,
        top_p = 1,
        extra_body={
            'enable_thinking': enable_thinking,
            'thinking_budget': thinking_budget,
            'top_k':32,
        },
    )

    # Extract reasoning content and answer content (for display)
    reasoning_content = ""
    answer_content = ""
    
    if response.choices:
        for choice in response.choices:
            message = choice.message
            if message:
                # Check if reasoning_content exists
                if hasattr(message, 'reasoning_content') and message.reasoning_content:
                    reasoning_content = message.reasoning_content
                    print("\n" + "=" * 20 + " Thinking Process " + "=" * 20 + "\n")
                    print(reasoning_content)
                
                # Extract answer content
                if message.content:
                    answer_content = message.content
                    print("\n" + "=" * 20 + " Complete Response " + "=" * 20 + "\n")
                    print(answer_content)
    
    # Convert entire response object to string
    response_str = str(response)
    
    # Return dictionary containing stringified response
    full_response = {
        "full_response_str": response_str,
        "extracted_reasoning": reasoning_content,
        "extracted_answer": answer_content,
        "request_prompt": full_prompt
    }
    
    return full_response

# --- Main Execution Logic ---
if __name__ == "__main__":
    import random
    
    # --- Configuration Parameters ---
    mapping_json_path = "./image_mask_mapping.json"
    guideline_file = "../annotation/Medical Deepfake Detection Guideline-v2.md" # Guideline file path
    image_size = 1024  # Image size
    total_samples = 2  # Number of samples to randomly select from all images

    # Read image_mask_mapping.json
    print(f"Reading mapping file: {mapping_json_path}")
    with open(mapping_json_path, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
    
    print(f"Found {len(mapping_data['mapping'])} image-mask pairs in total")
    
    # Randomly select total_samples samples from all images
    all_samples = mapping_data['mapping']
    if len(all_samples) >= total_samples:
        selected_samples = random.sample(all_samples, total_samples)
    else:
        selected_samples = all_samples  # If insufficient samples, select all
    
    print(f"\nRandomly selected {len(selected_samples)} samples:")
    for idx, sample in enumerate(selected_samples, 1):
        print(f"  {idx}. {Path(sample['image_path']).name} (Model: {sample['model']})")

    # Get Deepfake detection guidelines - now only load full content
    full_deepfake_guidelines_content = load_full_guidelines_content(guideline_file)
    print(f"\nLoaded full Deepfake detection guidelines content (partial):\n{full_deepfake_guidelines_content[:500]}...\n")

    # --- Gemini API Configuration ---
    gemini_api_key = os.getenv("GEMINI_API_KEY") # Recommended to get API key from environment variable
    if not gemini_api_key:
        print("Warning: GEMINI_API_KEY environment variable not set. Will skip Gemini API requests.")

    # --- Qwen API Configuration ---
    qwen_api_key = os.getenv("DASHSCOPE_API_KEY") # Recommended to get API key from environment variable
    # Below is the base_url for Beijing region. If using Singapore region models, replace base_url with: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    qwen_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if not qwen_api_key:
        print("Warning: DASHSCOPE_API_KEY environment variable not set. Will skip Qwen API requests.")

    # --- Unified Configuration Parameters ---
    enable_thinking = True  # Whether to enable thinking process
    thinking_budget = 256   # Token budget for thinking process

    # --- Lists for collecting all results ---
    all_gemini_results = []
    all_qwen_results = []

    # --- Inference for each selected sample ---
    for sample_idx, sample in enumerate(selected_samples, 1):
        image_path = sample['image_path']
        mask_path = sample['mask_path']
        model = sample['model']
        dataset_type = sample['dataset_type']
        
        print("\n" + "=" * 80)
        print(f"[{sample_idx}/{len(selected_samples)}] Processing sample")
        print(f"Image: {Path(image_path).name}")
        print(f"Mask: {Path(mask_path).name}")
        print(f"Model: {model}")
        print("=" * 80)
        
        # Check if files exist
        if not Path(image_path).exists():
            print(f"Warning: Image file '{image_path}' does not exist, skipping.")
            continue
        
        if not Path(mask_path).exists():
            print(f"Warning: Mask file '{mask_path}' does not exist, skipping.")
            continue

        # Extract bbox from mask
        print("Extracting bounding box from mask...")
        bbox_abs, bbox_normalized = extract_bbox_from_mask(mask_path, image_size)
        print(f"Absolute coordinate bbox: {bbox_abs} [xmin, ymin, xmax, ymax]")
        print(f"Normalized bbox (0-1000): {bbox_normalized} [ymin, xmin, ymax, xmax]")

        # Crop image using bbox
        # Determine save path: JSON files saved in current directory, cropped images also saved in current directory (next to JSON)
        image_name = Path(image_path).stem
        # cropped_image_path = f"./{image_name}_cropped.jpeg"
        
        # print(f"Cropping image...")
        # crop_image_with_bbox(image_path, bbox_abs, cropped_image_path)

        # --- Execute Requests ---
        
        # Gemini API Request
        if gemini_api_key:
            print("\n" + "#" * 30 + " Gemini API Request " + "#" * 30 + "\n")
            print("Analyzing image using Gemini API, please wait...")
            try:
                current_sample_guidelines = get_filtered_guidelines(full_deepfake_guidelines_content, dataset_type)
                gemini_response_obj, gemini_response_dict = analyze_deepfake_image_gemini(
                    image_path, bbox_normalized, gemini_api_key, current_sample_guidelines,
                    enable_thinking=enable_thinking, thinking_budget=thinking_budget
                )
                print("\n--- Gemini API Response ---")
                print(f"Reasoning content length: {len(gemini_response_dict.get('extracted_reasoning', ''))} characters")
                print(f"Answer content length: {len(gemini_response_dict.get('extracted_answer', ''))} characters")
                
                # Add image absolute path to result
                gemini_result = {
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "model": model,
                    "image_name": Path(image_path).name,
                    **gemini_response_dict
                }
                all_gemini_results.append(gemini_result)
                print(f"Gemini response added to results list")
            except Exception as e:
                print(f"Gemini API request failed: {e}")
        
        # Qwen API Request
        if qwen_api_key:
            print("\n" + "#" * 30 + " Qwen API Request " + "#" * 30 + "\n")
            print("Analyzing image using Qwen API, please wait...")
            try:
                current_sample_guidelines = get_filtered_guidelines(full_deepfake_guidelines_content, dataset_type)
                qwen_response = analyze_deepfake_image_qwen(
                    image_path, bbox_normalized, qwen_api_key, qwen_base_url, current_sample_guidelines,
                    enable_thinking=enable_thinking, thinking_budget=thinking_budget
                )
                print(f"\nReasoning content length: {len(qwen_response.get('extracted_reasoning', ''))} characters")
                print(f"Answer content length: {len(qwen_response.get('extracted_answer', ''))} characters")
                
                # Add image absolute path to result
                qwen_result = {
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "model": model,
                    "image_name": Path(image_path).name,
                    **qwen_response
                }
                all_qwen_results.append(qwen_result)
                print(f"Qwen response added to results list")
            except Exception as e:
                print(f"Qwen API request failed: {e}")
    
    # --- Save all results to merged JSON file ---
    print("\n" + "=" * 80)
    print("Saving all results to merged JSON file...")
    print("=" * 80)
    
    if all_gemini_results:
        gemini_merged_path = "./all_gemini_responses-grounded-v2.json"
        with open(gemini_merged_path, 'w', encoding='utf-8') as f:
            json.dump(all_gemini_results, f, ensure_ascii=False, indent=2)
        print(f"\nAll Gemini responses saved to: {gemini_merged_path}")
        print(f"Total {len(all_gemini_results)} records")
    
    if all_qwen_results:
        qwen_merged_path = "./all_qwen_responses-grounded-v2.json"
        with open(qwen_merged_path, 'w', encoding='utf-8') as f:
            json.dump(all_qwen_results, f, ensure_ascii=False, indent=2)
        print(f"\nAll Qwen responses saved to: {qwen_merged_path}")
        print(f"Total {len(all_qwen_results)} records")
    
    print("\n" + "=" * 80)
    print("All samples processed!")
    print("=" * 80)
