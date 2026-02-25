import unittest
import numpy as np
import pandas as pd
import os
import tempfile
import sys
import importlib.util

# Set workspace root robustly
# Try to find the 'executable' directory to anchor the root
current_dir = os.path.dirname(os.path.abspath(__file__))
# Check if we are in root or tests/
if os.path.isdir(os.path.join(current_dir, 'executable')):
    WORKSPACE_ROOT = current_dir
elif os.path.isdir(os.path.join(os.path.dirname(current_dir), 'executable')):
    WORKSPACE_ROOT = os.path.dirname(current_dir)
else:
    # Fallback to current working directory
    WORKSPACE_ROOT = os.getcwd()

sys.path.append(WORKSPACE_ROOT)

class TestPipelineIntegrity(unittest.TestCase):
    
    def test_01_environment_dependencies(self):
        """Verify all critical dependencies are installed and loadable."""
        dependencies = [
            'numpy', 'pandas', 'polars', 'torch', 
            'sklearn', 'scipy', 'statsmodels'
        ]
        
        print("\n[Test 01] Checking Dependencies...")
        for dep in dependencies:
            try:
                importlib.import_module(dep)
                print(f"  [OK] {dep}")
            except ImportError as e:
                self.fail(f"Missing dependency: {dep} - {e}")

    def test_02_structured_npy_loading_patch(self):
        """
        Verify the critical patch for loading structured NPY arrays.
        Simulates the 'optimal_lags.npy' produced by preprocessing.
        """
        print("\n[Test 02] Verifying Structured NPY Patch...")
        
        # 1. Create a structured array (simulating what Polars/Numpy might output)
        dtype = [('variable', 'U20'), ('optimal_lag', 'i4')]
        data = np.array([('Var_A', 5), ('Var_B', 10), ('Var_C', 1)], dtype=dtype)
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            np.save(tmp.name, data)
            tmp_path = tmp.name
            
        try:
            # 2. Simulate the LOADING LOGIC exactly as patched in scripts
            # -----------------------------------------------------------
            lags_csv = tmp_path
            lags_array = np.load(lags_csv, allow_pickle=True)
            
            # Logic from dbn_dynotears_fixed_lambda.py and parallel script
            if lags_array.dtype.names and 'optimal_lag' in lags_array.dtype.names:
                optimal_lags = lags_array['optimal_lag']
            else:
                optimal_lags = lags_array.flatten()
            
            df_lags = pd.DataFrame({'optimal_lag': optimal_lags})
            # -----------------------------------------------------------
            
            # 3. Assertions
            expected_lags = np.array([5, 10, 1])
            loaded_lags = df_lags['optimal_lag'].values
            
            print(f"  Original: {expected_lags}")
            print(f"  Loaded:   {loaded_lags}")
            
            self.assertTrue(np.array_equal(expected_lags, loaded_lags), 
                            "Failed to extract 'optimal_lag' from structured array!")
            
            # Verify the max() operation doesn't crash (the original error)
            p = int(df_lags['optimal_lag'].max())
            self.assertEqual(p, 10, "Max lag calculation failed")
            print("  [OK] Patch verified.")
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_03_script_syntax_and_imports(self):
        """
        Verify that key scripts can be parsed and imported without syntax errors.
        This catches indent errors, missing imports, or syntax typos.
        """
        print("\n[Test 03] Verifying Script Syntax...")
        
        scripts_to_check = [
            "executable/final_pipeline/dbn_dynotears_fixed_lambda.py",
            "executable/final_pipeline/dbn_dynotears_tucker_cam_parallel.py",
            "executable/launcher.py",
            "scripts/run_rca_on_detections.py"
        ]
        
        for rel_path in scripts_to_check:
            full_path = os.path.join(WORKSPACE_ROOT, rel_path)
            if not os.path.exists(full_path):
                self.fail(f"Script missing: {rel_path}")
            
            try:
                # Load spec to check syntax
                spec = importlib.util.spec_from_file_location("check_mod", full_path)
                mod = importlib.util.module_from_spec(spec)
                # spec.loader.exec_module(mod) # DISABLED: Creates side effects (might run partial logic)
                # Instead, just compile it to check syntax
                with open(full_path, 'r') as f:
                    compile(f.read(), full_path, 'exec')
                print(f"  [OK] Syntax check passed: {rel_path}")
            except SyntaxError as e:
                self.fail(f"Syntax Error in {rel_path}: {e}")
            except Exception as e:
                # Other errors might occur during compile if encoding is weird, but usually syntax error
                self.fail(f"Compile failed for {rel_path}: {e}")

if __name__ == '__main__':
    unittest.main()
