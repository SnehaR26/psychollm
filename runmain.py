import pandas as pd
import argparse
import csv
import sys
import io
import re
import os
from runtemp import runpipeline 

def main():
     # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the pipeline with command-line arguments.")
    parser.add_argument("--csv_file", metavar="csv_file",type=str, help="Path to the input CSV file")
    parser.add_argument("--scale_type", metavar="scale_type",type=int, choices=[1, 2], help="1 for standard Likert scale, 2 for reversed Likert scale")
    parser.add_argument("--prompt", metavar="prompt",type=int, choices=[1, 2], help="prompt function options")
    parser.add_argument("--resfile", metavar="resfile",type=str, help="Path to the output results file")
    parser.add_argument("--ptest", metavar="ptest",type=str, help="Type of psychometric test:- bfi or sd3")
    args = parser.parse_args()

    # Load CSV data and process as per arguments
    df1 = pd.read_csv(args.csv_file)
    df1 = df1.head()

    # Define the scales
    likert_scale = [
        "1 - Strongly Disagree",
        "2 - Disagree",
        "3 - Neutral",
        "4 - Agree",
        "5 - Strongly Agree"
    ]
    likert_scale_reversed = [
        "5 - Strongly Agree",
        "4 - Agree",
        "3 - Neutral",
        "2 - Disagree",
        "1 - Strongly Disagree"
    ]

    # Choose the appropriate scale
    selected_scale = likert_scale if args.scale_type == 1 else likert_scale_reversed
    
    # Fixed model name
    model = "tiiuae/falcon-7b-instruct"
    
    # Run the pipeline with provided arguments
    runpipeline(model, selected_scale, df1, args.prompt, args.ptest, args.resfile)

if __name__ == "__main__":
    main()
