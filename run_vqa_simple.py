#!/usr/bin/env python3
import json, subprocess, os, sys
from datetime import datetime

# load config.json
cfg = json.load(open("experiments_vit.json"))
BASE_DIR = cfg["base_output_dir"]
SLURM = cfg["slurm"]
SEEDS = cfg["seeds"]
EXPS  = cfg["experiments"]

os.makedirs(BASE_DIR, exist_ok=True)

# Create log file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(BASE_DIR, f"run_vqa_simple_log_{timestamp}.txt")

def log_print(message):
    """Print to console and write to log file"""
    print(message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")

# Initialize log file
with open(log_file, "w", encoding="utf-8") as f:
    f.write(f"VQA-Only Generation Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")

log_print("=== VQA-Only Generation ===")
log_print(f"Output dir: {BASE_DIR}")
log_print(f"Log file: {log_file}")
log_print(f"Partition: {SLURM['partition']}  GRES: {SLURM['gres']}  Node: {SLURM['node']}")
log_print(f"Seeds: {SEEDS}\n")

for exp in EXPS:
    name   = exp["name"]
    prompt = exp["prompt"]
    gs     = exp["guidance_scale"]
    neg    = exp.get("negative_prompt", "")
    topk   = exp.get("topk", 1)
    freq   = exp.get("freq", 1)
    oracle_id = exp.get("oracle_id", None)
    questions = exp.get("questions", [])
    start_t = exp.get("start_t", 0)
    end_t = exp.get("end_t", 28)
    main_object = exp.get("main_object", None)

    exp_dir = os.path.join(BASE_DIR, name)
    os.makedirs(exp_dir, exist_ok=True)

    log_print(f">>> Experiment: {name}")
    log_print(f"    prompt: {prompt}")
    log_print(f"    guidance_scale: {gs}   neg: '{neg}'   topk: {topk}   freq: {freq}")
    if questions:
        log_print(f"    questions: {questions}")
    log_print("")

    for seed in SEEDS:        
        log_print(f"[seed {seed}] Generating VQA-guided image...")
        
        # VQA-GUIDED GENERATION ONLY
        vqa_cmd = [
            "python", "gen_utils/generate_sd35_image.py",
            "--prompt", prompt,
            "--negative_prompt", neg,
            "--output_dir", exp_dir,
            "--seed", str(seed),
            # "--steps", str(end_t - start_t), TODO fixed to 28 currently
            "--guidance_scale", str(gs),
            # "--height", "1024", TODO fixed to 1024 currently
            # "--width", "1024", TODO fixed to 1024 currently
            "--topk", str(topk),
            "--freq", str(freq),
            "--vqa_start_step", str(start_t),
            "--vqa_stop_step", str(end_t),
        ]

        for q in questions:
            vqa_cmd += ["--question", q]

        # Only add oracle_id if it's not None
        if oracle_id is not None:
            vqa_cmd.extend(["--oracle_id", oracle_id])

        if main_object is not None:
            vqa_cmd += ["--main_object", str(main_object)]

        # Log the command being executed
        log_print(f"    Command: {' '.join(vqa_cmd)}")
        
        # Run command and capture output
        try:
            result = subprocess.run(vqa_cmd, capture_output=True, text=True)  # 1 hour timeout
            status = "✓" if result.returncode == 0 else "✗"
            log_print(f"   → {status} (return code: {result.returncode})")
            
            # Log stdout if available
            if result.stdout.strip():
                log_print("    STDOUT:")
                for line in result.stdout.strip().split('\n'):
                    log_print(f"      {line}")
            
            # Log stderr if available
            if result.stderr.strip():
                log_print("    STDERR:")
                for line in result.stderr.strip().split('\n'):
                    log_print(f"      {line}")
                    
        except subprocess.TimeoutExpired:
            log_print("   → ✗ (timeout after 1 hour)")
        except Exception as e:
            log_print(f"   → ✗ (error: {str(e)})")
        
        log_print("")

log_print("=== VQA generation complete! ===")
log_print(f"\nFull log saved to: {log_file}") 