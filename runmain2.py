import multiprocessing
import pandas as pd
import argparse
import csv
import sys
import io
import re
import os
from runtemp import runpipeline 
import multiprocessing
import time

if __name__ == "__main__":
    # Get the number of processes from the user
    num_processes = int(input("Enter the number of processes: "))
    
    # Initialize list for processes
    processes = []
    model = "tiiuae/falcon-7b-instruct"
    df1 = pd.read_csv("./Datasubs/good.csv")#args.csv_file
    df1 = df1.head()

    likert_scale = [
        "1 - Strongly Disagree",
        "2 - Disagree",
        "3 - Neutral",
        "4 - Agree",
        "5 - Strongly Agree"
    ]
    # For each process, get specific parameters and create a process instance
    for i in range(num_processes):
        param1 = input(f"Enter parameter1 data file for process {i+1}: ")
        param2 = input(f"Enter parameter2 res file for process {i+1}: ")
        #param3 = input(f"Enter parameter for process {i+1}: ")
        #resfile = f"paralleltest/file{i+1}.csv"  # Automatically name output files as file1.csv, file2.csv, etc.
        df1 = pd.read_csv(param1)#args.csv_file
        df1 = df1.head()
        # Create a new process for each parameter and output file
        process = multiprocessing.Process(target=runpipeline, args=(model,likert_scale,df1,1,"bfi", param2))
        processes.append(process)  # Append the process to the list
    
    # Start all processes
    for process in processes:
        process.start()
    
    # Wait for all processes to complete
    for process in processes:
        process.join()

    print("All processes completed.")
