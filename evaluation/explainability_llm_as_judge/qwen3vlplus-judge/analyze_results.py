import os
import json
import glob
from collections import defaultdict

def analyze_results():
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found at {results_dir}")
        return

    # Find all .jsonl files
    jsonl_files = glob.glob(os.path.join(results_dir, "*.jsonl"))
    
    if not jsonl_files:
        print("No result files found.")
        return

    model_stats = {}

    for file_path in jsonl_files:
        model_name = os.path.basename(file_path).replace("results_", "").replace(".jsonl", "")
        
        scores = {
            "logical_correctness": [],
            "visual_hallucination": [],
            "medical_professionalism": []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    judge_results = data.get("judge_results", {})
                    
                    for criteria in scores.keys():
                        score_val = judge_results.get(criteria, {}).get("score")
                        if score_val is not None:
                            scores[criteria].append(score_val)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # Calculate averages for this model
        if any(scores.values()):
            model_stats[model_name] = {}
            total_sum = 0
            total_count = 0
            
            for criteria, vals in scores.items():
                if vals:
                    avg = sum(vals) / len(vals)
                    model_stats[model_name][criteria] = {
                        "avg": avg,
                        "count": len(vals)
                    }
                    total_sum += sum(vals)
                    total_count += len(vals)
                else:
                    model_stats[model_name][criteria] = {"avg": 0, "count": 0}
            
            model_stats[model_name]["overall"] = total_sum / total_count if total_count > 0 else 0
            model_stats[model_name]["sample_count"] = len(next(iter(scores.values()))) # Assume all criteria have same count

    # Generate Report
    print("\n" + "="*85)
    print(f"{'Model Name':<20} | {'Logical':<8} | {'Visual':<8} | {'Medical':<8} | {'Overall':<8} | {'Samples':<7}")
    print("-" * 85)
    
    # Sort models by overall score descending
    sorted_models = sorted(model_stats.items(), key=lambda x: x[1]["overall"], reverse=True)
    
    for model, stats in sorted_models:
        print(f"{model:<20} | "
              f"{stats['logical_correctness']['avg']:<8.2f} | "
              f"{stats['visual_hallucination']['avg']:<8.2f} | "
              f"{stats['medical_professionalism']['avg']:<8.2f} | "
              f"{stats['overall']:<8.2f} | "
              f"{stats['sample_count']:<7}")
    
    print("="*85 + "\n")

    # Save to report file
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "summary_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Gemini 3 Pro LLM-as-Judge Summary Report\n")
        f.write("="*85 + "\n")
        f.write(f"{'Model Name':<20} | {'Logical':<8} | {'Visual':<8} | {'Medical':<8} | {'Overall':<8} | {'Samples':<7}\n")
        f.write("-" * 85 + "\n")
        for model, stats in sorted_models:
            f.write(f"{model:<20} | "
                  f"{stats['logical_correctness']['avg']:<8.2f} | "
                  f"{stats['visual_hallucination']['avg']:<8.2f} | "
                  f"{stats['medical_professionalism']['avg']:<8.2f} | "
                  f"{stats['overall']:<8.2f} | "
                  f"{stats['sample_count']:<7}\n")
        f.write("="*85 + "\n")
    
    print(f"Summary report saved to: {report_path}")

if __name__ == "__main__":
    analyze_results()

