import pandas as pd
import os
import random

def sample_csv(filename):
    # Ensure the file exists
    if not os.path.isfile(filename):
        print(f"Error: File '{filename}' not found.")
        return
    
    # Read the CSV file
    df = pd.read_csv(filename)
    
    # Ensure there are enough rows to sample from
    if len(df) < 1001:
        print("Error: The file must have at least 1001 rows.")
        return
    
    # Take the first row and 1000 random rows
    sampled_df = pd.concat([df.iloc[:1], df.iloc[random.sample(range(1, len(df)), 1000)]], ignore_index=True)
    
    # Generate output filename
    name, ext = os.path.splitext(filename)
    output_filename = f"{name}_1000{ext}"
    
    # Save the sampled data
    sampled_df.to_csv(output_filename, index=False)
    print(f"Sampled data saved to '{output_filename}'.")

if __name__ == "__main__":
    filename = input("Enter the CSV filename: ").strip()
    sample_csv(filename)
