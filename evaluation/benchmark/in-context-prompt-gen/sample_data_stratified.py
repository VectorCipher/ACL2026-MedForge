import json
import random
import re
import os
from collections import defaultdict

# Configuration
INPUT_FILE = '../../data/sft_train_dataset.json'
OUTPUT_FILE = './master_sample.json'
STATS_FILE = './sampling_stats.txt'

TARGET_REAL = 20
TARGET_FAKE = 40

def extract_evidence(text):
    match = re.search(r'<evidence>(.*?)</evidence>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def parse_modality_from_type(type_str):
    if 'chest-xray' in type_str:
        return 'Chest X-ray'
    elif 'brain-mri' in type_str:
        return 'Brain MRI'
    elif 'fundus' in type_str:
        return 'Fundus'
    return type_str

def sample_data():
    print(f"Loading {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    real_pool = []
    fake_pool = defaultdict(list) # Key: (model, type)

    print("Categorizing data...")
    for idx, item in enumerate(data):
        # Basic validation
        if 'images' not in item or not item['images']:
            continue
        
        image_path = item['images'][0]
        
        # Get assistant content
        assistant_content = ""
        for msg in item.get('messages', []):
            if msg['role'] == 'assistant':
                assistant_content = msg['content']
                break
        
        if not assistant_content:
            continue
            
        evidence = extract_evidence(assistant_content)
        if not evidence:
            continue

        # Determine Class & Metadata
        if '/real/' in image_path:
            # Path: .../real/{modality}/...
            try:
                parts = image_path.split('/real/')
                subpath = parts[1]
                modality_slug = subpath.split('/')[0]
                
                modality = modality_slug
                if 'chest-xray' in modality_slug: modality = 'Chest X-ray'
                elif 'brain-mri' in modality_slug: modality = 'Brain MRI'
                elif 'fundus' in modality_slug: modality = 'Fundus'
                
                entry = {
                    "id": idx,
                    "class": "Real",
                    "modality": modality,
                    "type": "N/A",
                    "model": "N/A",
                    "evidence": evidence,
                    "path": image_path
                }
                real_pool.append(entry)
            except:
                continue
                
        elif '/deepfake/' in image_path:
            # Path: .../deepfake/{model}/{type}/...
            try:
                parts = image_path.split('/deepfake/')
                subpath = parts[1]
                path_parts = subpath.split('/')
                if len(path_parts) < 2:
                    continue
                    
                model = path_parts[0]
                type_slug = path_parts[1]
                
                modality = parse_modality_from_type(type_slug)
                
                manipulation = "Unknown"
                if 'remove' in type_slug:
                    manipulation = "Lesion Removal"
                elif 'edit' in type_slug:
                    manipulation = "Lesion Implant"
                
                entry = {
                    "id": idx,
                    "class": "Fake",
                    "modality": modality,
                    "type": type_slug, # Kept for internal filtering (remove/edit)
                    "manipulation": manipulation, # Display name
                    "model": model,
                    "evidence": evidence,
                    "path": image_path
                }
                fake_pool[(model, type_slug)].append(entry)
            except:
                continue

    print(f"Pool Sizes: Real={len(real_pool)}, Fake Groups={len(fake_pool)}")

    # Sampling Real
    selected_real = random.sample(real_pool, min(len(real_pool), TARGET_REAL))
    
    # Sampling Fake (Stratified)
    selected_fake = []
    fake_groups = list(fake_pool.keys())
    
    if not fake_groups:
        print("No fake groups found!")
        return

    # Calculate base samples per group
    # We have ~20 groups (10 models * 2 types). 40 / 20 = 2 samples per group.
    # If groups > 40, we sample 1 from random subset.
    
    samples_per_group = max(1, TARGET_FAKE // len(fake_groups))
    remainder = TARGET_FAKE
    
    # First pass
    for key in fake_groups:
        group_items = fake_pool[key]
        k = min(len(group_items), samples_per_group)
        picked = random.sample(group_items, k)
        selected_fake.extend(picked)
        remainder -= len(picked)
    
    # Second pass (fill remainder)
    if remainder > 0:
        # Flatten all remaining unpicked items
        remaining_pool = []
        picked_ids = {x['id'] for x in selected_fake}
        
        for key in fake_groups:
            for item in fake_pool[key]:
                if item['id'] not in picked_ids:
                    remaining_pool.append(item)
        
        if remaining_pool:
            extras = random.sample(remaining_pool, min(len(remaining_pool), remainder))
            selected_fake.extend(extras)

    # Trim if over (unlikely with this logic unless min > target)
    if len(selected_fake) > TARGET_FAKE:
        selected_fake = selected_fake[:TARGET_FAKE]

    final_set = selected_real + selected_fake
    random.shuffle(final_set)
    
    print(f"Selected {len(final_set)} items ({len(selected_real)} Real, {len(selected_fake)} Fake).")

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_set, f, indent=2)
    print(f"Saved to {OUTPUT_FILE}")
    
    # Generate Stats Report
    stats_lines = []
    stats_lines.append(f"Total Samples: {len(final_set)}")
    stats_lines.append(f"Real: {len(selected_real)}")
    stats_lines.append(f"Fake: {len(selected_fake)}")
    
    model_counts = defaultdict(int)
    type_counts = defaultdict(int)
    
    for item in selected_fake:
        model_counts[item['model']] += 1
        type_counts[item['type']] += 1
        
    stats_lines.append("\nBy Model:")
    for m, c in sorted(model_counts.items()):
        stats_lines.append(f"  {m}: {c}")
        
    stats_lines.append("\nBy Type:")
    for t, c in sorted(type_counts.items()):
        stats_lines.append(f"  {t}: {c}")
        
    with open(STATS_FILE, 'w') as f:
        f.write('\n'.join(stats_lines))
    print(f"Stats saved to {STATS_FILE}")

if __name__ == "__main__":
    sample_data()

