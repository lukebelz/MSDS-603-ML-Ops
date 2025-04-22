import os
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Split World Happiness data into train and test sets')
    parser.add_argument('--input_path', type=str, default='../data/world_happiness_report.csv',
                        help='Path to the raw CSV data')
    parser.add_argument('--output_dir', type=str, default='../data',
                        help='Directory to save the train and test CSV files')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction of data to use for testing (default: 0.2)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define output paths
    train_output_path = os.path.join(args.output_dir, 'world_happiness_train_data.csv')
    test_output_path = os.path.join(args.output_dir, 'world_happiness_test_data.csv')
    
    # Load the data
    print(f"Loading data from {args.input_path}...")
    df = pd.read_csv(args.input_path)
    print(f"Loaded {len(df)} rows")
    
    # Split the data
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed
    )
    
    print(f"Split data into {len(train_df)} training samples and {len(test_df)} test samples")
    
    # Save the splits
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)
    
    print(f"Saved training data to {train_output_path}")
    print(f"Saved test data to {test_output_path}")
    print("Data splitting complete!")

if __name__ == '__main__':
    main()