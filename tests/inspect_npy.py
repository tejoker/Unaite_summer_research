import numpy as np
import sys
import os

def inspect(file_path):
    print(f"Inspecting: {file_path}")
    if not os.path.exists(file_path):
        print("File not found!")
        return

    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"Type: {type(data)}")
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
        
        if data.size > 0:
            print("First item:", data[0])
            print("First item type:", type(data[0]))
            
        if data.dtype.names:
            print("Field names:", data.dtype.names)
            if 'optimal_lag' in data.dtype.names:
                col = data['optimal_lag']
                print("optimal_lag column sample:", col[:5])
                print("optimal_lag col dtype:", col.dtype)
        else:
            print("No field names (Flat or Object array).")
            # Check if it's an object array of tuples
            if data.dtype == 'O':
                print("It is an OBJECT array.")
                print("Sample content:", data[0])
                
    except Exception as e:
        print(f"Error loading: {e}")

if __name__ == "__main__":
    path = "/home/nicolas_b/program_internship_paul_wurth/results/SMD_machine-1-6_golden_baseline/run_000/preprocessing/machine-1-6_optimal_lags.npy"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    inspect(path)
