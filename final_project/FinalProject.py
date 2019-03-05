import os
import pandas as pd
import platform

marvel_file_name = 'marvel-wikia-data.csv'
dc_file_name = 'dc-wikia-data.csv'

if platform.system() == 'Windows':
    home_dir = os.path.join('C:', os.path.sep, 'Users', 'jimmy_000')  # MS Windows home directory
else:  # Assumes Linux
    home_dir = os.path.join(os.path.sep + 'home', 'jgoddard')  # Linux home directory
input_dir = os.path.join(home_dir, 'src', 'git', 'CS677', 'final_project', 'datasets', 'fivethirtyeight')

marvel_input_file = os.path.join(input_dir, marvel_file_name)
marvel_df = pd.read_csv(marvel_input_file)

dc_input_file = os.path.join(input_dir, dc_file_name)
dc_df = pd.read_csv(dc_input_file)
