#!/usr/bin/env python3
"""
Verify Convergence Stability of Tucker-CAM.
Running this script produces the metrics for Appendix F.
"""

import os
import sys
import numpy as np
import subprocess
from pathlib import Path
import statistics

def main():
    print("="*80)
    print("TUCKER-CAM CONVERGENCE VERIFICATION")
    print("="*80)
    
    workspace_root = Path(__file__).resolve().parent.parent
    launcher = workspace_root / "executable" / "launcher.py"
    data_file = workspace_root / "telemanom" / "test_dataset_merged_clean.csv" # Sample data
    output_base = workspace_root / "results" / "convergence_test"
    
    n_runs = 10 # 50 takes too long for a quick check, user can increase
    losses = []
    
    print(f"Running {n_runs} independent trials...")
    
    for i in range(n_runs):
        seed = 1000 + i
        run_dir = output_base / f"run_{i}"
        
        # We need to capture the *loss* from the log output
        cmd = [
            sys.executable, str(launcher),
            "--data", str(data_file),
            "--output", str(run_dir),
            "--seed", str(seed),
            "--skip-steps", "preprocessing" # Reuse preproc if possible (launcher handles this check)
        ]
        
        # We need to parse stdout for final loss
        # Since launcher directs output to stdout, we capture it
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Simple parsing: find last "loss="
            # Log format: "Iter 99: loss=0.0423..."
            lines = result.stdout.splitlines()
            final_loss = None
            for line in reversed(lines):
                if "loss=" in line:
                    parts = line.split()
                    for p in parts:
                        if p.startswith("loss="):
                            final_loss = float(p.split("=")[1].strip(","))
                            break
                    if final_loss is not None:
                        break
            
            if final_loss is not None:
                losses.append(final_loss)
                print(f"Run {i}: Loss = {final_loss:.6f}")
            else:
                print(f"Run {i}: Loss not found (check logs)")
                
        except subprocess.CalledProcessError as e:
            print(f"Run {i}: FAILED")
            
    if losses:
        mean_loss = statistics.mean(losses)
        std_loss = statistics.stdev(losses) if len(losses) > 1 else 0.0
        print("-" * 80)
        print(f"Results over {len(losses)} runs:")
        print(f"Mean Loss: {mean_loss:.6f}")
        print(f"Std Loss:  {std_loss:.6e}")
        print("-" * 80)
        
        if std_loss < 1e-4:
            print("✓ CONVERGENCE STABLE (Std < 1e-4)")
        else:
            print("⚠ CONVERGENCE UNSTABLE")

if __name__ == "__main__":
    main()
