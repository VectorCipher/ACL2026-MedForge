# Medical Deepfake Detection In-Context Prompt Generator

This tool is used to extract evidence from training datasets and generate three different types of in-context learning prompts for medical deepfake detection tasks.

## Overview

Generates prompts for three detection scenarios:
- **In Domain**: Comprehensive detection features covering all models and manipulation types.
- **Cross Model**: Summary of features after excluding specific models (e.g., Nano-Banana/Gemini, GPT, SD3.5-Med, XL-Inpaint).
- **Cross Forgery**: Features only for Lesion Removal (excluding Implant/Edit).

## File Descriptions

| File | Description |
|------|-------------|
| `sample_data_stratified.py` | Data sampling script: extracts representative samples from the training set. |
| `generate_prompts.py` | Prompt generation script: uses Gemini 3 Pro API to generate prompts. |
| `master_sample.json` | 60 sampled data items (40 Fake + 20 Real). |
| `sampling_stats.txt` | Statistics about the sampled data. |
| `generated_prompts.txt` | **Final Output**: The three generated prompt texts. |
| `exclusion_report.txt` | Exclusion report: details what was excluded and their corresponding features. |

## Steps to Use

### 1. Environment Preparation

Ensure necessary Python packages are installed:
```bash
pip install google-genai
```

Set your Gemini API Key:
```bash
export GEMINI_API_KEY="your_api_key_here"
```

### 2. Data Sampling (Optional)

If `master_sample.json` does not exist or you need to re-sample:

```bash
python3 sample_data_stratified.py
```

**Sampling Configuration**:
- Samples from `../../data/sft_train_dataset.json`.
- Target: 60 samples (40 Fake + 20 Real).
- Covers all models and manipulation types (Remove/Edit).
- Each sample includes complete metadata: class, modality, manipulation, model, and evidence.

**Output**:
- `master_sample.json`: The sampled data.
- `sampling_stats.txt`: Statistical information.

### 3. Generate Prompts

Run the generation script:

```bash
python3 generate_prompts.py
```

**Configuration Details**:
- Model used: `gemini-3-pro-preview`.
- Thinking Level: `high` (highest reasoning depth).
- Generates detailed and comprehensive output.

**Output**:
- `generated_prompts.txt`: The three generated prompt texts.
- `exclusion_report.txt`: Exclusion analysis report.

## Output Format

### generated_prompts.txt

Contains three sections:

```
=== In Domain ===
[Comprehensive detection features, covering all models and manipulation types]

=== Cross Model ===
[Summary of features after excluding specific models]

=== Cross Forgery ===
[Feature descriptions for Lesion Removal only]
```

### exclusion_report.txt

Contains two exclusion analyses:
- **Cross Model Exclusion Analysis**: Unique features of excluded models (Gemini, GPT, SD3.5-Med, XL-Inpaint).
- **Cross Forgery Exclusion Analysis**: Unique features of Lesion Implant.

## Customization

### Change Sample Size

Edit `sample_data_stratified.py`:
```python
TARGET_REAL = 20   # Number of Real samples
TARGET_FAKE = 40   # Number of Fake samples
```

### Change Excluded Models

Edit `generate_prompts.py`:
```python
EXCLUDED_MODELS = {
    'gemini', 
    'gpt', 
    'stable-diffusion-3.5-medium', 
    'stable-diffusion-xl-1.0-inpainting-0.1'
}
```

### Change Input Data Path

Edit `sample_data_stratified.py`:
```python
INPUT_FILE = '../../data/sft_train_dataset.json'
```

## Important Notes

1. **API Costs**: Using the Gemini 3 Pro API will incur costs. Ensure your account has sufficient balance.
2. **Data Paths**: Ensure the `sft_train_dataset.json` path is correct.
3. **API Key**: The `GEMINI_API_KEY` environment variable must be set.
4. **Generation Time**: Since `thinking_level="high"` is used, generation may take several minutes.

## Troubleshooting

### Error: GEMINI_API_KEY not set
- Check if the environment variable is correctly set: `echo $GEMINI_API_KEY`.

### Error: File not found
- Check if `master_sample.json` exists, or run `sample_data_stratified.py` first.

### Error: API Call Failed
- Check if your API Key is valid.
- Check your internet connection.
- Check if your API quota has been reached.

## Example Output

The generated prompts can be directly used for:
- In-context learning for MLLMs.
- Medical deepfake detection experiments.
- Evaluation of generalization across models and manipulation types.

## Support

If you encounter issues, please check:
1. File paths are correct.
2. API Key is valid.
3. Python environment and dependencies are complete.
