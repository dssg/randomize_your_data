# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 12:38:36 2018

@author: JoanWang
"""
import pandas as pd
import numpy as np
from column_randomization import randomize

def generate_data():
    '''
    Generate fake data (100 rows x 5 columns) valued between -10 and 10
    Outcome variable y is sum of each row, with random normal error inserted
    
    Outputs:
        Output files will be generated: fake_data.csv and fake_data_no_actual.csv
    '''
    print("Generating fake data...")
    df = pd.DataFrame(np.random.randint(-10,10, size=(100,5)), 
                 columns=['x1', 'x2', 'x3', 'x4', 'x5'])
    df['pred_y'] = df.sum(axis = 1)
    df['actual_y'] = df['pred_y'].apply(lambda x: x + np.random.normal())
    df.index.name = 'idx'
    df.to_csv('fake_data.csv')
    
    df_no_actual = df.drop(['pred_y', 'actual_y'], axis = 1)
    df_no_actual.to_csv('fake_data_no_actual.csv')
    return

def test_randomization(file, file_with_outcome, index_col = None):
    # Before randomization
    orig_df = pd.read_csv(file_with_outcome, index_col = index_col)
    orig_df['sq_error'] = (orig_df['actual_y'] - orig_df['pred_y'])**2
    orig_mse =  np.sum(orig_df['sq_error']) / orig_df.shape[0]
    print('original mean squared error = ' + str(orig_mse))
    
    # After randomization
    random_df = randomize(file, index_col)
    random_df = random_df.join(orig_df['actual_y'], how='inner')
    random_df['pred_y'] = random_df[['x1', 'x2', 'x3', 'x4', 'x5']].sum(axis = 1)
    
    random_df['sq_error'] = (random_df['actual_y'] - random_df['pred_y'])**2
    random_mse =  np.sum(random_df['sq_error']) / random_df.shape[0]
    print('random mean squared error = ' + str(random_mse))
    
    return random_df

if __name__ == "__main__":
    generate_data()
    file = 'fake_data_no_actual.csv'
    file_with_outcome = 'fake_data.csv'
    df = test_randomization(file, file_with_outcome, 'idx') 
   