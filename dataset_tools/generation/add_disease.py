#!/usr/bin/env python3
"""
Add Disease: Edit normal images to disease images
Supports multi-threading, resume from checkpoint, API failure retry
"""

from google import genai
from google.genai import types
from pathlib import Path
from PIL import Image
from io import BytesIO
import json
import argparse
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


class DiseaseAdder:
    def __init__(self, dataset_path, output_base, max_workers=20, max_rounds=5, test_limit=None):
        """
        Initialize disease adder
        
        Args:
            dataset_path: Dataset path (e.g., full-data/MIMIC_single_disease_selection_dim1024_1k_per_class)
            output_base: Base output path (e.g., img_gen)
            max_workers: Number of concurrent threads
            max_rounds: Maximum number of attempt rounds
            test_limit: Test mode, only process specified number of images (None means process all)
        """
        self.dataset_path = Path(dataset_path)
        self.dataset_name = self.dataset_path.name
        self.output_base = Path(output_base)
        self.max_workers = max_workers
        self.max_rounds = max_rounds
        self.test_limit = test_limit
        
        # Output directories
        self.output_dir = self.output_base / f"{self.dataset_name}-edit"
        self.failed_dir = self.output_base / f"{self.dataset_name}-edit-failed"
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress and failure records
        self.progress_file = self.output_dir / "progress.json"
        self.api_failures_file = self.output_dir / "api_failures.json"
        self.failed_summary_file = self.failed_dir / "failed_summary.json"
        self.final_prompts_file = self.output_dir / "final_prompts.json"
        self.conversations_file = self.output_dir / "all_conversations.json"
        
        # Load progress
        self.progress = self.load_progress()
        self.api_failures = self.load_api_failures()
        self.failed_summary = self.load_failed_summary()
        self.final_prompts = self.load_final_prompts()
        self.all_conversations = self.load_all_conversations()
        
        # Thread lock
        self.lock = threading.Lock()
        
    def load_progress(self):
        """Load progress"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_progress(self):
        """Save progress (thread-safe)"""
        with self.lock:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
    
    def load_api_failures(self):
        """Load API failure records"""
        if self.api_failures_file.exists():
            with open(self.api_failures_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_api_failures(self):
        """Save API failure records (thread-safe)"""
        with self.lock:
            with open(self.api_failures_file, 'w') as f:
                json.dump(self.api_failures, f, indent=2, ensure_ascii=False)
    
    def load_failed_summary(self):
        """Load failed task summary"""
        if self.failed_summary_file.exists():
            with open(self.failed_summary_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_failed_summary(self):
        """Save failed task summary (thread-safe)"""
        with self.lock:
            with open(self.failed_summary_file, 'w') as f:
                json.dump(self.failed_summary, f, indent=2, ensure_ascii=False)
    
    def load_final_prompts(self):
        """Load final prompts"""
        if self.final_prompts_file.exists():
            with open(self.final_prompts_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_final_prompts(self):
        """Save final prompts (thread-safe)"""
        with self.lock:
            with open(self.final_prompts_file, 'w') as f:
                json.dump(self.final_prompts, f, indent=2, ensure_ascii=False)
    
    def load_all_conversations(self):
        """Load all conversation records"""
        if self.conversations_file.exists():
            with open(self.conversations_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_all_conversations(self):
        """Save all conversation records (thread-safe)"""
        with self.lock:
            with open(self.conversations_file, 'w') as f:
                json.dump(self.all_conversations, f, indent=2, ensure_ascii=False)
    
    def cleanup_api_failures(self):
        """Clean up api_failed records in progress, prepare for retry"""
        retry_count = 0
        for task_key, status in list(self.progress.items()):
            if status == "api_failed":
                del self.progress[task_key]
                retry_count += 1
        if retry_count > 0:
            self.save_progress()
        return retry_count
    
    def api_call_with_retry(self, func, max_retries=3):
        """API call with retry"""
        for attempt in range(max_retries):
            try:
                return func(), None
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    time.sleep(wait_time)
                else:
                    return None, str(e)
        return None, "Max retries exceeded"
    
    def extract_thinking_and_content(self, response):
        """Extract thinking process and content"""
        thinking = ""
        content = ""
        
        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                thinking += part.text + "\n"
            else:
                content += part.text + "\n"
        
        return thinking.strip(), content.strip()
    
    def generate_initial_prompt(self, disease_name, dataset_type):
        """Generate initial editing prompt"""
        system_instruction = f"""You are a medical image editing expert. Generate a clear, concise prompt to edit a normal {dataset_type} image to show {disease_name}.

Requirements:
1. Easy for non-medical personnel to understand
2. Must look natural and realistic, not detectable as deepfake
3. Should follow common sense and medical plausibility
4. Be specific about what to add/modify

CRITICAL CONSTRAINTS - Fidelity Preservation:
- Preserve original image noise, grain texture, and contrast
- Do NOT alter device artifacts, watermarks, grid lines, or annotations
- Keep the same imaging characteristics (brightness, sharpness, field of view)

CRITICAL CONSTRAINTS - Negative Rules:
- NO adding text, labels, or annotations
- NO sharp unnatural edges or boundaries
- NO repetitive/duplicated structures
- NO non-target pathologies
- NO obvious deepfake artifacts

CRITICAL CONSTRAINTS - Minimal Change Principle (Counterfactual Minimality):
- ONLY modify areas directly related to {disease_name}
- Keep all other anatomical structures, textures, and background UNCHANGED
- Minimal intervention: change as little as possible to show the disease

The prompt should be concise and short.
Add a clear warning: do not edit any element other than adding the disease feature. Keep everything else in the image exactly the same,
preserving the original style, lighting, and composition.

Add a clear warning at both the beginning and the end of the prompt: do not edit any element other than adding the disease feature. Keep everything else in the image exactly the same,
preserving the original style, lighting, and composition.

Return ONLY the editing prompt in English, no explanations."""
        
        client = genai.Client()
        
        def call():
            return client.models.generate_content(
                model="gemini-2.5-pro",
                contents=system_instruction,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=True,
                    )
                )
            )
        
        response, error = self.api_call_with_retry(call)
        if error:
            return None, None, error
        
        thinking, prompt = self.extract_thinking_and_content(response)
        return thinking, prompt, None
    
    def update_prompt(self, original_image_path, disease_name, prompt_history, dataset_type):
        """
        Update editing prompt
        
        Args:
            original_image_path: Original image path
            disease_name: Disease name
            prompt_history: History prompt list, format: [{"round": 1, "prompt": "...", "verification": {...}}, ...]
            dataset_type: Dataset type
        """
        with open(original_image_path, 'rb') as f:
            image_bytes = f.read()
        
        # Build history failure records
        history_text = ""
        for i, history in enumerate(prompt_history, 1):
            history_text += f"""
Attempt {i}:
  Prompt: {history['prompt']}
  Verification Result:
    - Has disease: {history['verification']['has_disease']}
    - Structure reasonable: {history['verification']['structure_reasonable']}
    - Looks realistic: {history['verification']['looks_realistic']}
    - Reason: {history['verification']['reason']}
"""
        
        system_instruction = f"""You are a medical image editing expert. Multiple previous editing attempts have failed. You need to analyze ALL previous attempts and generate a BETTER prompt.

HISTORY OF ALL PREVIOUS ATTEMPTS:
{history_text}

Looking at the ORIGINAL normal image and analyzing the patterns of failures above, generate an IMPROVED editing prompt to add {disease_name} to this {dataset_type} image.

ANALYSIS REQUIREMENTS:
1. Identify common issues across multiple attempts
2. Learn from what didn't work in previous rounds
3. Avoid repeating the same mistakes
4. Address ALL verification issues mentioned in the history

BASIC REQUIREMENTS:
1. Easy for non-medical personnel to understand
2. Must look natural and realistic, not detectable as deepfake
3. Should follow common sense and medical plausibility
4. Be specific about what to add/modify

CRITICAL CONSTRAINTS - Fidelity Preservation:
- Preserve original image noise, grain texture, and contrast
- Do NOT alter device artifacts, watermarks, grid lines, or annotations
- Keep the same imaging characteristics (brightness, sharpness, field of view)

CRITICAL CONSTRAINTS - Negative Rules:
- NO adding text, labels, or annotations
- NO sharp unnatural edges or boundaries
- NO repetitive/duplicated structures
- NO non-target pathologies
- NO obvious deepfake artifacts

CRITICAL CONSTRAINTS - Minimal Change Principle (Counterfactual Minimality):
- ONLY modify areas directly related to {disease_name}
- Keep all other anatomical structures, textures, and background UNCHANGED
- Minimal intervention: change as little as possible to show the disease

The prompt should be concise and short.
Add a clear warning: do not edit any element other than adding the disease feature. Keep everything else in the image exactly the same,
preserving the original style, lighting, and composition.

Return ONLY the editing prompt in English, no explanations."""
        
        client = genai.Client()
        
        def call():
            return client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
                    system_instruction
                ],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=True,
                    )
                )
            )
        
        response, error = self.api_call_with_retry(call)
        if error:
            return None, None, error
        
        thinking, prompt = self.extract_thinking_and_content(response)
        return thinking, prompt, None
    
    def edit_image(self, image_path, edit_prompt):
        """Edit image"""
        image = Image.open(image_path)
        
        client = genai.Client()
        
        def call():
            return client.models.generate_content(
                model="gemini-2.5-flash-image-preview",
                contents=[edit_prompt, image]
            )
        
        response, error = self.api_call_with_retry(call)
        if error:
            return None, error
        
        # Extract generated image
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                edited_image = Image.open(BytesIO(part.inline_data.data))
                return edited_image, None
        
        return None, "No image generated"
    
    def verify_edited_image(self, original_image_path, edited_image, disease_name, dataset_type):
        """Verify edited image (compare original and edited images)"""
        # Read original image
        with open(original_image_path, 'rb') as f:
            original_img_bytes = f.read()
        
        # Convert edited image to bytes
        edited_img_byte_arr = BytesIO()
        edited_image.save(edited_img_byte_arr, format='JPEG')
        edited_img_bytes = edited_img_byte_arr.getvalue()
        
        verification_instruction = f"""You are a medical image verification expert. You are given TWO images:
1. ORIGINAL IMAGE (normal): A normal {dataset_type} image without disease
2. EDITED IMAGE: The same image after editing to add {disease_name}

Your task is to verify the editing quality by COMPARING the two images side-by-side.

CRITICAL VERIFICATION - Minimal Change Principle:
Compare the ORIGINAL and EDITED images carefully. The editing should ONLY add the disease-related features. Check:
- Are non-disease areas (background, other anatomical structures, imaging artifacts, watermarks, annotations) IDENTICAL between original and edited?
- Are there unnecessary changes to regions that should remain untouched?
- Does the edited version preserve the exact same imaging characteristics (noise, grain, contrast, brightness) in unchanged areas?

IMPORTANT: Take your time to think carefully. Medical image editing is challenging, and minor imperfections are acceptable as long as the overall goal is achieved. Be thoughtful and balanced in your evaluation - don't reject an image for trivial issues.

Check these aspects by comparing both images:
1. Has disease: Does the EDITED image show signs of {disease_name}? (should be True for diseased)
   - Compare with ORIGINAL to identify what was added
   - Check if disease features are visible and recognizable in EDITED
   - Check if the disease features are added in the correct location

2. Structure reasonable: Are the anatomical structures in EDITED image reasonable and correct?
   - Compare anatomical structures between ORIGINAL and EDITED
   - Check if editing preserved anatomical correctness
   - Check if the anatomical structures are added in the correct location
   - Check if the anatomical structures are added in the correct size and shape
   - Check if the anatomical structures are added in the correct color
   - Check if the anatomical structures are added in the correct texture
   - Check if the anatomical structures are added in the correct contrast

3. Looks realistic: Does the EDITED image look like a real medical image?
   - Compare the texture, noise, and visual quality with ORIGINAL
   - Some editing traces are inevitable; focus on whether it could pass as a real medical image to non-experts

4. Minimal changes preserved: By comparing ORIGINAL vs EDITED side-by-side:
   - Are changes limited ONLY to disease addition areas?
   - Are all other regions (background, non-diseased anatomy, artifacts, text, watermarks) UNCHANGED?
   - Is the overall image fidelity preserved (same noise pattern, contrast, brightness in unchanged areas)?
   - This is CRITICAL - excessive changes to non-disease areas should result in rejection

Additional verification for image fidelity (compare ORIGINAL vs EDITED):
- Check if unchanged areas preserve the exact same noise/grain texture
- Check if there are unnatural sharp edges or boundaries in edited regions
- Check if there are added text, repetitive structures, or deepfake artifacts
- Check if modifications are truly minimal (reject if too many areas changed unnecessarily)

Return your evaluation in this JSON format:
{{
    "qualified": true/false,
    "has_disease": true/false,
    "structure_reasonable": true/false,
    "looks_realistic": true/false,
    "minimal_changes_preserved": true/false,
    "reason": "detailed explanation comparing ORIGINAL vs EDITED (mention: what changed, what stayed the same, whether changes were appropriate and minimal, both strengths and weaknesses; explain your reasoning for acceptance or rejection)"
}}

Only qualified if:
- has_disease is TRUE (disease successfully added)
- structure_reasonable is TRUE (anatomical structures correct)
- looks_realistic is TRUE (appears as real medical image)
- minimal_changes_preserved is TRUE (only disease areas changed, other areas unchanged)
- No MAJOR fidelity issues detected"""
        
        client = genai.Client()
        
        def call():
            return client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    "ORIGINAL IMAGE (normal):",
                    types.Part.from_bytes(data=original_img_bytes, mime_type='image/jpeg'),
                    "EDITED IMAGE (after disease addition):",
                    types.Part.from_bytes(data=edited_img_bytes, mime_type='image/jpeg'),
                    verification_instruction
                ],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=True,
                        thinking_budget=32768  # Maximum thinking depth
                    )
                )
            )
        
        response, error = self.api_call_with_retry(call)
        if error:
            return None, None, error
        
        thinking, content = self.extract_thinking_and_content(response)
        
        # Parse JSON
        try:
            # Extract JSON
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            result = json.loads(content)
            return thinking, result, None
        except json.JSONDecodeError as e:
            return thinking, None, f"JSON parse error: {str(e)}"
    
    def get_dataset_type(self):
        """Get dataset type description"""
        name = self.dataset_name.lower()
        if 'mimic' in name or 'chest' in name:
            return "chest X-ray"
        elif 'brain' in name or 'tumor' in name:
            return "brain MRI"
        elif 'odir' in name or 'fundus' in name:
            return "fundus"
        return "medical"
    
    def process_single_task(self, normal_image_path, disease_name):
        """Process single editing task"""
        image_name = Path(normal_image_path).stem
        task_key = f"{image_name}-{disease_name}"
        
        # Check if already completed
        if task_key in self.progress:
            return {"status": "skipped", "task": task_key}
        
        dataset_type = self.get_dataset_type()
        
        # Conversation records
        conversation = {
            "image": Path(normal_image_path).name,
            "disease": disease_name,
            "status": "failed",
            "rounds": []
        }
        
        current_prompt = None
        
        for round_num in range(1, self.max_rounds + 1):
            round_data = {"round": round_num}
            
            # 1. Generate or update prompt
            if round_num == 1:
                thinking, prompt, error = self.generate_initial_prompt(disease_name, dataset_type)
                if error:
                    # API failure - record and mark as processed (skip)
                    with self.lock:
                        self.api_failures.append({
                            "task": task_key,
                            "step": "generate_prompt",
                            "round": round_num,
                            "error": error,
                            "image": Path(normal_image_path).name,
                            "disease": disease_name
                        })
                        self.progress[task_key] = "api_failed"
                    self.save_api_failures()
                    self.save_progress()
                    return {"status": "api_failed", "task": task_key, "error": error}
                
                round_data["generate_prompt"] = {
                    "thinking_summary": thinking,
                    "prompt": prompt
                }
                current_prompt = prompt
            else:
                # Build complete history for LLM analysis
                prompt_history = []
                for prev_round in conversation["rounds"]:
                    if "generate_prompt" in prev_round and "verification" in prev_round:
                        prompt_history.append({
                            "round": prev_round["round"],
                            "prompt": prev_round["generate_prompt"]["prompt"],
                            "verification": prev_round["verification"]
                        })
                
                # Update prompt using complete history
                thinking, prompt, error = self.update_prompt(
                    normal_image_path, disease_name, prompt_history, dataset_type
                )
                if error:
                    # API failure - record and mark as processed (skip)
                    with self.lock:
                        self.api_failures.append({
                            "task": task_key,
                            "step": "update_prompt",
                            "round": round_num,
                            "error": error,
                            "image": Path(normal_image_path).name,
                            "disease": disease_name
                        })
                        self.progress[task_key] = "api_failed"
                    self.save_api_failures()
                    self.save_progress()
                    return {"status": "api_failed", "task": task_key, "error": error}
                
                round_data["generate_prompt"] = {
                    "thinking_summary": thinking,
                    "prompt": prompt
                }
                current_prompt = prompt
            
            # 2. Edit image
            edited_image, error = self.edit_image(normal_image_path, current_prompt)
            if error:
                round_data["edit_result"] = {"success": False, "error": error}
                conversation["rounds"].append(round_data)
                
                # API失败 - 记录并标记为已处理（跳过）
                with self.lock:
                    self.api_failures.append({
                        "task": task_key,
                        "step": "edit_image",
                        "round": round_num,
                        "error": error,
                        "image": Path(normal_image_path).name,
                        "disease": disease_name
                    })
                    self.progress[task_key] = "api_failed"
                self.save_api_failures()
                self.save_progress()
                return {"status": "api_failed", "task": task_key, "error": error}
            
            round_data["edit_result"] = {"success": True}
            
            # 3. Verify image (pass original image path for comparison)
            thinking, verification, error = self.verify_edited_image(normal_image_path, edited_image, disease_name, dataset_type)
            if error:
                round_data["verification"] = {"error": error}
                conversation["rounds"].append(round_data)
                
                # API失败 - 记录并标记为已处理（跳过）
                with self.lock:
                    self.api_failures.append({
                        "task": task_key,
                        "step": "verify_image",
                        "round": round_num,
                        "error": error,
                        "image": Path(normal_image_path).name,
                        "disease": disease_name
                    })
                    self.progress[task_key] = "api_failed"
                self.save_api_failures()
                self.save_progress()
                return {"status": "api_failed", "task": task_key, "error": error}
            
            round_data["verification"] = verification
            round_data["verification"]["thinking_summary"] = thinking
            conversation["rounds"].append(round_data)
            
            # 4. Check if qualified
            if verification.get("qualified", False):
                # Success! Save image
                output_subdir = self.output_dir / disease_name
                output_subdir.mkdir(parents=True, exist_ok=True)
                output_path = output_subdir / f"{image_name}-{disease_name}-edit.jpeg"
                edited_image.save(output_path, 'JPEG', quality=95)
                
                conversation["status"] = "success"
                conversation["final_prompt"] = current_prompt
                conversation["final_image_path"] = str(output_path)
                
                # Save to centralized conversation records
                with self.lock:
                    self.all_conversations[task_key] = conversation
                self.save_all_conversations()
                
                # Save final prompt
                with self.lock:
                    self.final_prompts[task_key] = {
                        "image": Path(normal_image_path).name,
                        "disease": disease_name,
                        "status": "success",
                        "final_prompt": current_prompt,
                        "rounds": round_num
                    }
                self.save_final_prompts()
                
                # Update progress
                with self.lock:
                    self.progress[task_key] = "success"
                self.save_progress()
                
                return {"status": "success", "task": task_key, "rounds": round_num}
            else:
                # Save failed intermediate images
                failed_subdir = self.failed_dir / disease_name
                failed_subdir.mkdir(parents=True, exist_ok=True)
                failed_path = failed_subdir / f"{image_name}-{disease_name}-edit-failed-{round_num}.jpeg"
                edited_image.save(failed_path, 'JPEG', quality=95)
        
        # All rounds failed - image only saved in failed directory
        conversation["status"] = "failed"
        conversation["final_prompt"] = current_prompt
        conversation["final_image_path"] = str(self.failed_dir / disease_name / f"{image_name}-{disease_name}-edit-failed-{self.max_rounds}.jpeg")
        
        # Save to centralized conversation records
        with self.lock:
            self.all_conversations[task_key] = conversation
        self.save_all_conversations()
        
        # Save to failure summary (including complete conversation history)
        with self.lock:
            self.failed_summary.append({
                "task": task_key,
                "image": Path(normal_image_path).name,
                "disease": disease_name,
                "final_prompt": current_prompt,
                "rounds": self.max_rounds,
                "last_verification": conversation["rounds"][-1]["verification"],
                "final_image_path": conversation["final_image_path"],
                "full_conversation": conversation  # Save complete conversation history
            })
        self.save_failed_summary()
        
        # Save final prompt
        with self.lock:
            self.final_prompts[task_key] = {
                "image": Path(normal_image_path).name,
                "disease": disease_name,
                "status": "failed",
                "final_prompt": current_prompt,
                "rounds": self.max_rounds
            }
        self.save_final_prompts()
        
        # Update progress (mark as processed to avoid duplicates)
        with self.lock:
            self.progress[task_key] = "failed"
        self.save_progress()
        
        return {"status": "failed", "task": task_key, "rounds": self.max_rounds}
    
    def get_all_tasks(self):
        """Get all pending tasks"""
        tasks = []
        
        # Find normal image directory
        normal_dir = self.dataset_path / "normal"
        if not normal_dir.exists():
            raise ValueError(f"Normal directory not found: {normal_dir}")
        
        # Find all disease categories
        disease_dirs = [d for d in self.dataset_path.iterdir() 
                       if d.is_dir() and d.name != "normal"]
        disease_names = [d.name for d in disease_dirs]
        
        # Find all normal images
        normal_images = list(normal_dir.glob("*.jpeg")) + list(normal_dir.glob("*.jpg"))
        
        # Generate editing tasks for all diseases for each normal image
        for normal_img in normal_images:
            for disease in disease_names:
                tasks.append((str(normal_img), disease))
        return tasks
    
    def run(self):
        """Run complete workflow"""
        # API failure retry logic
        if hasattr(self, 'retry_mode') and self.retry_mode:
            retry_count = self.cleanup_api_failures()
            if retry_count > 0:
                print(f"\n✓ Cleaned {retry_count} api_failed records, ready for retry\n")
            else:
                print(f"\n✓ Retry mode: No api_failed records found\n")
        
        tasks = self.get_all_tasks()
        
        # If in test mode, only process specified number of tasks
        if self.test_limit is not None:
            tasks = tasks[:self.test_limit]

        
        print(f"\n{'='*80}")
        print(f"Add Disease - {self.dataset_name}")
        print(f"{'='*80}")
        print(f"Dataset path: {self.dataset_path}")
        print(f"Output directory: {self.output_dir}")
        if self.test_limit is not None:
            print(f"Test mode: Only processing first {self.test_limit} tasks")
        print(f"Total tasks: {len(tasks)}")
        print(f"Completed: {len(self.progress)}")
        print(f"Pending: {len(tasks) - len(self.progress)}")
        print(f"Concurrent threads: {self.max_workers}")
        print(f"Maximum rounds: {self.max_rounds}")
        print(f"{'='*80}\n")
        
        # Statistics
        stats = {
            "success": 0,
            "failed": 0,
            "api_failed": 0,
            "skipped": 0
        }
        
        # Multi-threaded processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.process_single_task, img_path, disease): (img_path, disease)
                for img_path, disease in tasks
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                try:
                    result = future.result()
                    stats[result["status"]] += 1
                except Exception as e:
                    print(f"\nTask exception: {e}")
                    stats["api_failed"] += 1
        
        # Print final statistics
        print(f"\n{'='*80}")
        print(f"Processing complete!")
        print(f"{'='*80}")
        print(f"Success: {stats['success']}")
        print(f"Failed: {stats['failed']}")
        print(f"API failed: {stats['api_failed']}")
        print(f"Skipped: {stats['skipped']}")
        print(f"\nConversation records: {self.conversations_file}")
        print(f"Final prompts: {self.final_prompts_file}")
        print(f"Failure summary: {self.failed_summary_file}")
        if self.api_failures:
            print(f"API failure records: {self.api_failures_file}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Add Disease: Edit normal images to disease images')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., MIMIC_single_disease_selection_dim1024_1k_per_class, brain-tumor-1024x1024, ODIR-5K-classified-1024x1024)')
    parser.add_argument('--max-workers', type=int, default=20,
                       help='Number of concurrent threads (default: 20)')
    parser.add_argument('--max-rounds', type=int, default=5,
                       help='Maximum attempt rounds (default: 5)')
    parser.add_argument('--test-limit', type=int, default=None,
                       help='Test mode: Only process specified number of images (default: None, process all)')
    parser.add_argument('--retry', action='store_true',
                       help='Retry api_failed tasks (automatically clean failure records and rerun)')
    
    args = parser.parse_args()
    
    # Path settings
    BASE_DIR = Path("./data")
    dataset_path = BASE_DIR / args.dataset
    
    
    # Output directory: Use current script directory
    OUTPUT_BASE = Path(__file__).parent
    if not dataset_path.exists():
        print(f"Error: Dataset does not exist {dataset_path}")
        return
    
    # Create processor
    adder = DiseaseAdder(
        dataset_path=dataset_path,
        output_base=OUTPUT_BASE,
        max_workers=args.max_workers,
        max_rounds=args.max_rounds,
        test_limit=args.test_limit
    )
    
    # Pass retry mode parameter
    adder.retry_mode = args.retry
    
    # Run
    adder.run()


if __name__ == '__main__':
    main()


