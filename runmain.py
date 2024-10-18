import pandas as pd
import csv
import sys
import io
import re
import os
from runtemp import runpipeline 

def main():
    df1=pd.read_csv('Datasubs/good.csv')
    df1=df1.head()
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
    resfile='./F7B_res/results_test.csv'
    model="tiiuae/falcon-7b-instruct"
    runpipeline(model,'1',likert_scale,df1,1,resfile)
if __name__ == "__main__":
    main()
