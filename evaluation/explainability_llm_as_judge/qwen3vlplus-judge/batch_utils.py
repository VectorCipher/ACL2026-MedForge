import base64
import os
import json
import re
from PIL import Image, ImageDraw
import io

def remap_image_path(remote_path):
    """
    Map remote paths to local paths for dataset access.
    Prioritize using the MED_FORGE_DATASET_PATH environment variable.
    """
    # Placeholder for remote path mapping logic if needed
    if "/data/" in remote_path:
        # Example logic: extract relative path after a common prefix
        parts = remote_path.split("/data/")
        if len(parts) > 1:
            relative_path = parts[-1]
            base_path = os.getenv("MED_FORGE_DATASET_PATH")
            if base_path:
                return os.path.join(base_path, relative_path)
    
    # Fallback to absolute path or current directory
    if os.path.exists(remote_path):
        return remote_path
        
    # Attempt to find within the workspace data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_data = os.path.abspath(os.path.join(script_dir, "../../../data"))
    
    # Try to find the image filename in the local data directory
    image_filename = os.path.basename(remote_path)
    # Search recursively for the filename in the data directory
    for root, dirs, files in os.walk(project_root_data):
        if image_filename in files:
            return os.path.join(root, image_filename)
            
    return remote_path

def parse_model_response(response_text):
    """
    Parse the model response to extract bounding box and description.
    
    Returns:
        dict: {
            'bbox': [x1, y1, x2, y2] (integers, normalized 0-1000) or None,
            'explanation': str (description text)
        }
    """
    result = {'bbox': None, 'explanation': ""}
    
    # Extract explanation (after "Description:")
    desc_match = re.search(r"Description:\s*(.*)", response_text, re.DOTALL)
    if desc_match:
        result['explanation'] = desc_match.group(1).strip()
    
    # Extract bbox from <box ... />
    # Format: <box class="deepfake" x1="850" y1="430" x2="965" y2="600" />
    box_match = re.search(r'<box\s+class="deepfake"\s+x1="(\d+)"\s+y1="(\d+)"\s+x2="(\d+)"\s+y2="(\d+)"\s*/>', response_text)
    if box_match:
        x1, y1, x2, y2 = map(int, box_match.groups())
        result['bbox'] = [x1, y1, x2, y2]
        
    return result

def draw_bbox_and_encode(image_path, bbox_norm, save_debug_path=None):
    """
    Draw bbox on image and return base64 string.
    Handles errors by clamping coordinates or using a default bbox if invalid.
    
    Args:
        image_path: Path to local image
        bbox_norm: [x1, y1, x2, y2] normalized to 0-1000, or None
        save_debug_path: If provided, save the modified image to this path
        
    Returns:
        str: Base64 encoded image (with data URL prefix for Qwen/OpenAI)
    """
    local_path = remap_image_path(image_path)
    
    if not os.path.exists(local_path):
        # Fail gracefully if file really doesn't exist
        print(f"Image not found: {local_path}")
        return None
        
    try:
        with Image.open(local_path) as img:
            img = img.convert("RGB") # Ensure RGB
            width, height = img.size
            
            # 1. Handle missing/invalid bbox
            if not bbox_norm or not isinstance(bbox_norm, list) or len(bbox_norm) != 4:
                # Fallback: Draw a box around the entire image or center
                print(f"Warning: Invalid bbox {bbox_norm} for {local_path}. Using default full-image bbox.")
                bbox_norm = [0, 0, 1000, 1000]

            # 2. Clamp coordinates and ensure valid box
            def clamp(val, min_v, max_v):
                return max(min_v, min(val, max_v))
            
            x1 = clamp(int(bbox_norm[0]), 0, 1000)
            y1 = clamp(int(bbox_norm[1]), 0, 1000)
            x2 = clamp(int(bbox_norm[2]), 0, 1000)
            y2 = clamp(int(bbox_norm[3]), 0, 1000)
            
            # Swap if inverted
            if x1 > x2: x1, x2 = x2, x1
            if y1 > y2: y1, y2 = y2, y1
            
            # Ensure at least 1 pixel width/height to be visible
            if x1 == x2: 
                if x2 < 1000: x2 += 1 
                else: x1 -= 1
            if y1 == y2:
                if y2 < 1000: y2 += 1
                else: y1 -= 1
            
            # Denormalize
            abs_x1 = int(x1 / 1000 * width)
            abs_y1 = int(y1 / 1000 * height)
            abs_x2 = int(x2 / 1000 * width)
            abs_y2 = int(y2 / 1000 * height)
            
            # Draw bbox
            draw = ImageDraw.Draw(img)
            draw.rectangle([abs_x1, abs_y1, abs_x2, abs_y2], outline="red", width=5)
            
            # Save debug image if requested
            if save_debug_path:
                try:
                    img.save(save_debug_path)
                except Exception as e:
                    print(f"Warning: Could not save debug image: {e}")
            
            # Convert to base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            base64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{base64_data}"
            
    except Exception as e:
        print(f"Error processing image {local_path}: {e}")
        # Fallback: return original image
        try:
            with open(local_path, "rb") as f:
                base64_data = base64.b64encode(f.read()).decode("utf-8")
                return f"data:image/jpeg;base64,{base64_data}"
        except:
            return None

def encode_image_without_bbox(image_path, save_debug_path=None):
    """
    Encode image to base64 without drawing any bbox (for real images).
    
    Args:
        image_path: Path to local image
        save_debug_path: If provided, save the image to this path
        
    Returns:
        str: Base64 encoded image (with data URL prefix for Qwen/OpenAI)
    """
    local_path = remap_image_path(image_path)
    
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Image not found: {local_path}")
        
    try:
        with Image.open(local_path) as img:
            img = img.convert("RGB")
            
            # Save debug image if requested
            if save_debug_path:
                img.save(save_debug_path)
            
            # Convert to base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            base64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{base64_data}"
            
    except Exception as e:
        print(f"Error processing image {local_path}: {e}")
        return None

def build_multi_model_judge_prompt(responses, reference_answer=None, image_type="fake"):
    """
    Build the LLM-as-Judge prompt for evaluating multiple models against medical forensic guidelines.
    
    Args:
        responses: Dict mapping model IDs (A, B, C, ...) to their response text.
        reference_answer: The ground truth explanation (from 'labels').
        image_type: "fake" or "real".
    """
    
    # Build responses text
    responses_text = ""
    for label in sorted(responses.keys()):
        text = responses[label]
        responses_text += f"Assistant {label}:\n{text}\n\n"
    
    # Build list of assistants for the prompt
    assistant_list = ", ".join(sorted(responses.keys()))
    num_assistants = len(responses)

    gt_section = ""
    if reference_answer:
        gt_section = f"""
### Ground Truth Information
**Ground Truth Explanation**: 
"{reference_answer}"

(Use this as the definitive reference for what is actually present or manipulated in the image.)
"""

    if image_type == "fake":
        task_description = "The provided image contains a **RED BOUNDING BOX** indicating the ground truth location of an AI-generated manipulation (Deepfake)."
    else:
        task_description = "The provided image is an **AUTHENTIC (REAL)** medical image."

    # Build output format section dynamically
    output_format_lines = []
    for label in sorted(responses.keys()):
        output_format_lines.append(f"{label}. Score: [1-5]\nReasoning: [Short forensic justification]")
    output_format = "\n".join(output_format_lines)

    return f"""# Role
You are a distinguished Medical Image Forensics Expert acting as an impartial judge. Your expertise covers Radiology (MRI, CXR) and Ophthalmology (Fundus), specifically in identifying AI-generated (Deepfake) anomalies versus real pathological features.

# Task
Please examine the provided image attentively and serve as an unbiased judge in assessing the quality of responses from {num_assistants} AI assistants. You will evaluate how well each assistant identifies and explains the forensic nature of the image based on professional medical imaging standards.

{task_description}

# Input Data
1. **Visual Evidence**: 
   - Image 1: Original medical image.
   - Image 2: Image with a red bounding box marking the ground truth deepfake region (only relevant if the image is a deepfake).
2. **Assistant Responses**: Responses from Assistants {assistant_list}.

{gt_section}

### Assistant Responses to Evaluate:
{responses_text}

# Evaluation Criteria (Medical Deepfake Detection Guidelines)
You must evaluate the responses based on these specific forensic criteria:

1. **Biological Plausibility & Secondary Effects**:
    - **Mass Effect**: Real lesions displace surrounding tissue. Check if the assistant identifies the presence or absence of compression, displacement, or deformation of adjacent structures.
    - **Host Reaction**: Real pathology triggers reactions like edema or inflammation. Check if the assistant notes the presence/absence of surrounding edema or infiltration.
    - **Chronological Inconsistency**: Diseases follow a timeline. Check if the assistant identifies if features appear out of order.

2. **Image Physics & Texture Consistency**:
    - **The "Sticker" Artifact**: AI lesions often have unnaturally sharp boundaries.
    - **Noise Distribution**: Medical images have inherent grain. AI regions may be smoother or have different texture patterns (inpainting artifacts).
    - **Vascular/Structural Continuity**: In Fundus or MRI, check for "floating" vessels, abrupt endings, or anatomically incorrect branching.

3. **Modality-Specific Logic**:
    - **Brain MRI**: Gyral/Sulcal morphology, midline shift, multi-sequence consistency (T1/T2/FLAIR).
    - **Fundus**: Vascular tapering, Optic Disc/Macula positioning, hemorrhage depth (flame-shaped vs dot/blot).
    - **CXR**: 3D projection, lung markings behind the heart, rib continuity, secondary signs of collapse or congestion.

# Scoring Rubric (1-5 Scale)
- **Excellent (5)**: Perfect adherence to instructions. Excels in relevance, accuracy, and forensic depth. Correctly identifies key anomalies using valid forensic logic (e.g., mass effect, texture artifacts) and matches the Ground Truth perfectly.
- **Good (4)**: Well-aligned and accurate. Demonstrates a nuanced understanding and detailed granularity, but might lack the absolute forensic precision of an 'Excellent' response.
- **Average (3)**: Adequately addresses the inquiry with fair accuracy. Reflects basic forensic logic but may lack sophistication or depth in identifying subtle AI artifacts.
- **Fair (2)**: Partially addresses instructions but with evident shortcomings in accuracy or relevance. Shows superficial understanding or minor hallucinations.
- **Poor (1)**: Significantly deviates or fails to address the query. Lacks relevance, accuracy, and forensic reasoning.

# Output Format
Please provide your assessment in the following format:
{output_format}

Be objective. Do not favor certain names or positions. Do not allow response length to influence your evaluation.
"""

def build_benchmark_messages(image_path, prompt, base64_image=None):
    """
    Build messages in OpenAI Vision format
    """
    if not base64_image:
        # Fallback if base64_image is not provided directly
        with open(remap_image_path(image_path), "rb") as image_file:
            base64_data = base64.b64encode(image_file.read()).decode("utf-8")
            base64_image = f"data:image/jpeg;base64,{base64_data}"
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image
                    }
                }
            ]
        }
    ]
    return messages
