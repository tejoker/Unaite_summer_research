import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# CONFIG
ENTITIES = [
    "machine-1-1", "machine-1-2", "machine-1-3", "machine-1-4", "machine-1-5", "machine-1-6", "machine-1-7", "machine-1-8",
    "machine-2-1", "machine-2-2", "machine-2-3", "machine-2-4", "machine-2-5", "machine-2-6", "machine-2-7", "machine-2-8", "machine-2-9",
    "machine-3-1", "machine-3-2", "machine-3-3", "machine-3-4", "machine-3-5", "machine-3-6", "machine-3-7", "machine-3-8", "machine-3-9", "machine-3-10", "machine-3-11"
]
MAX_WORKERS = 3
PROJECT_ROOT = "/home/nicolas_b/program_internship_paul_wurth"
RESULTS_ROOT = "/mnt/disk2/results"
LOG_DIR = os.path.join(PROJECT_ROOT, "logs", f"benchmark_v9_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

def run_entity(entity):
    entity_log_path = os.path.join(LOG_DIR, f"{entity}.log")
    cmd = [
        "env",
        f"RESULTS_ROOT={RESULTS_ROOT}",
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "./scripts/run_entity_pipeline.sh",
        entity
    ]
    
    print(f"[{datetime.now()}] START {entity} -> Logs: {entity_log_path}")
    
    with open(entity_log_path, "w") as log_file:
        try:
            # Run subprocess with checks
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False  # Don't throw exception, check returncode manually
            )
            
            status = "SUCCESS" if result.returncode == 0 else f"FAILED({result.returncode})"
            print(f"[{datetime.now()}] FINISH {entity}: {status}")
            return result.returncode == 0
            
        except Exception as e:
            print(f"[{datetime.now()}] CRASH {entity}: {e}")
            log_file.write(f"\nLAUNCHER EXCEPTION: {e}\n")
            return False

def main():
    print(f"Starting V9 Benchmark (Python Orchestrator)")
    print(f"Entities: {len(ENTITIES)}")
    print(f"Parallelism: {MAX_WORKERS}")
    print(f"Logs: {LOG_DIR}")
    
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    
    # Ensure script is executable
    subprocess.run(["chmod", "+x", "./scripts/run_entity_pipeline.sh"], cwd=PROJECT_ROOT)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(run_entity, ENTITIES))
        
    print(f"All tasks completed.")
    success_count = sum(results)
    print(f"Success: {success_count}/{len(ENTITIES)}")

if __name__ == "__main__":
    main()
