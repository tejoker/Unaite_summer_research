import numpy as np
import pandas as pd
import tempfile
import os

def test_structured_array_loading():
    print("Test 1: Creating dummy structured array (Simulating preprocessing_no_mi.py output)...")
    
    # Create a structured array directly, similar to what might come from polished Polars->Numpy conversion
    # or just a complex save.
    # The error 'ufunc isnan not supported' suggests Pandas failed to interpret the object.
    
    # Define a structured dtype
    dt = np.dtype([('variable', 'U20'), ('optimal_lag', 'i4')])
    data = np.array([('Var1', 3), ('Var2', 5), ('Var3', 1)], dtype=dt)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "test_lags.npy")
        np.save(input_file, data)
        print(f"Saved structured array to {input_file}")
        
        # Now simulate the LOADING logic from the patched script
        print("Attempting to load...")
        try:
            lags_array = np.load(input_file, allow_pickle=True)
            
            # --- THE PATCHED LOGIC START ---
            if lags_array.dtype.names and 'optimal_lag' in lags_array.dtype.names:
                print("  -> Detected structured array with 'optimal_lag' field.")
                optimal_lags = lags_array['optimal_lag']
            else:
                print("  -> Flat array detection (fallback).")
                optimal_lags = lags_array.flatten()
                
            df_lags = pd.DataFrame({'optimal_lag': optimal_lags})
            # --- THE PATCHED LOGIC END ---
            
            print("Successfully created DataFrame:")
            print(df_lags)
            
            # Verify max calculation (the operation that crashed)
            p = int(df_lags['optimal_lag'].max()) if not df_lags.empty else 1
            print(f"Max lag (p): {p}")
            assert p == 5
            print("Test PASSED!")
            
        except Exception as e:
            print(f"Test FAILED with error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_structured_array_loading()
