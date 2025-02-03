import pandas as pd
import numpy as np

"""
    Create a spreadsheet where the columns corresponds to the relevant inputs,
    and the rows are the timelags sampled from a uniform distribution with
    min and max values are estimated by a random forest approach, plus and
    minus 6 hours.
"""

# NSG post processes data location
file = 'timelags_RandomForest.xlsx'

# Training df
t_df = pd.read_excel(file, sheet_name='timelags')
timelags_df = pd.DataFrame()

# Up to 6 hours from the mean timelag
N_samples = 20

# Initialise dict with the first input
minimum = np.min(t_df[t_df.columns[0]] - 6)
maximum = np.max(t_df[t_df.columns[0]] + 6)

d = {t_df.columns[0]:np.random.randint(minimum, maximum, N_samples)}

for i in range(1, len(t_df.columns)):
    minimum = np.min(t_df[t_df.columns[i]] - 6)
    maximum = np.max(t_df[t_df.columns[i]] + 7)
    d[t_df.columns[i]] = np.random.randint(minimum,
                                           maximum,
                                           N_samples)
    
samples = pd.DataFrame(d)

# Define an Excel writer object and the target file
samples.to_csv("timelag_samples.csv")
# writer = pd.ExcelWriter('Timelag_samples.xlsx')

# # Save to spreadsheet
# samples.to_excel(writer, sheet_name='time_lags', index=False)
# writer.save()