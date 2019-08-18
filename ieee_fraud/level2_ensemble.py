import numpy as np
import pandas as pd

#Base DF
submission_df = pd.read_csv('../../data/sample_submission.csv')
submission_df = submission_df.drop(['isFraud'],axis=1)

output_filenames = ['sample_submission']
for file in output_filenames:
    # Load file and rename isFraud with the filename
    temp_df = pd.read_csv('../../data/'+file+'.csv').rename(columns={"TransactionID":"TransactionID","isFraud":file})
    submission_df = submission_df.set_index('TransactionID').join(temp_df.set_index('TransactionID'))

# Arithmetic mean
mean = submission_df.mean(axis=1)
submission_df['arith_mean'] = mean

# Write submission based on sample file
def write_submission(df,col_name,filename):
    sample_submission = pd.read_csv('../../data/sample_submission.csv', index_col='TransactionID')
    sample_submission['isFraud'] = df[col_name]
    sample_submission.to_csv(''+filename)

write_submission(submission_df,'arith_mean','arith_mean_submission.csv')