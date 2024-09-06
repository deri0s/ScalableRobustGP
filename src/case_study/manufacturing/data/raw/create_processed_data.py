import pandas as pd
import numpy as np
from case_study.manufacturing.data.raw import data_processing_methods as dpm
from sklearn.preprocessing import MinMaxScaler as minmax

"""
    ISRA-5D:
    We assume that the data related to the coating process have been
    successfully rectified, and hence we do not ignore any points when
    assembling the training dataset.
"""

# ----------------------------------------------------------------------------
# USER OPTIONS
# ----------------------------------------------------------------------------

# Choose the scanner to read data
scanner = 'ISRA'

# Furnace properties that are planned (i.e. do not become uncertain as we look
# beyond their corresponding time lags)
planned_inputs = ['10425 Calculated Cullet Ratio',
                  '10091 Furnace Load',
                  '9400 Port 2 Gas Flow (SP)']

# Model inputs to retain (provided by Pilkington, excluding
# combination signals).
to_retain = ['2922 Closed Bottom Temperature - Downstream Working End (PV)',
             '2923 Filling Pocket Closed Bottom Temperature Centre (PV)',
             '2918 Closed Bottom Temperature - Port 6 (PV)',
             '2921 Closed Bottom Temperature - Upstream Working End (PV)',
             '10091 Furnace Load',
             '7546 Open Crown Temperature - Port 1 (PV)',
             '7746 Open Crown Temperature - Port 2 (PV)',
             '7522 Open Crown Temperature - Port 4 (PV)',
             '7483 Open Crown Temperature - Port 6 (PV)',
             '10271 C9 (T012) Upstream Refiner']

'''
# Combined combustion air flow signals
to_retain.append('Port 1 Combustion Air Flow (combined)')
to_retain.append('Port 2 - 3 Combustion Air Flow (combined)')
to_retain.append('Port 4 - 5 Combustion Air Flow (combined)')
to_retain.append('Port 6 - 7 Combustion Air Flow (combined)')
to_retain.append('Port 8 Combustion Air Flow (combined)')

# Combined front wall temperatures
to_retain.append('200000 Front Wall Temperature Average (PV)')

# Combined regenerator crown temperatures
to_retain.append('Regenerator Crown Temperature Port 2 (abs. difference)')
to_retain.append('Regenerator Crown Temperature Port 4 (abs. difference)')
to_retain.append('Regenerator Crown Temperature Port 6 (abs. difference)')
to_retain.append('Regenerator Crown Temperature Port 8 (abs. difference)')

# Combined regenerator base temperatures
to_retain.append('Regenerator Base Temperature Port 1 (abs. difference)')
to_retain.append('Regenerator Base Temperature Port 2 (abs. difference)')
to_retain.append('Regenerator Base Temperature Port 3 (abs. difference)')
to_retain.append('Regenerator Base Temperature Port 4 (abs. difference)')
to_retain.append('Regenerator Base Temperature Port 5 (abs. difference)')
to_retain.append('Regenerator Base Temperature Port 6 (abs. difference)')
to_retain.append('Regenerator Base Temperature Port 7 (abs. difference)')
to_retain.append('Regenerator Base Temperature Port 8 (abs. difference)')

# Other model inputs we can choose to retain if we want
to_retain.append('6539 Bath Pressure (PV)')
to_retain.append('11282 Chimney Draught Pressure - After Ecomomiser')
to_retain.append('5134 EEDS Average Gross Width Measurement')
to_retain.append('11213 Essential Services Board Cat \'B\' Supply - MV53')
to_retain.append('30208 Feeder Speed Measurement Left')
to_retain.append('5288 Front Wall Cooling Air Flow (PV)')
to_retain.append('11201 Furnace & Services Pack Sub S8 L.H. Section - HV13')
to_retain.append('11174 Furnace Bottom Temperature 18m D/S of B8')
to_retain.append('7474 Lehr Drive Line Shaft Speed')
to_retain.append('6463 Main Gas Pressure (PV)')
to_retain.append('11211 MV51')
to_retain.append('7443 Outside Ambient Temperature Measurement')
to_retain.append('7999 Outside Windspeed Anemometer')
to_retain.append('11105 Port 1 Combustion Air Flow LHS (OP)')
to_retain.append('11108 Port 1 Combustion Air Flow RHS (OP)')
to_retain.append('11111 Port 2 - 3 Combustion Air Flow LHS (OP)')
to_retain.append('11114 Port 2 - 3 Combustion Air Flow RHS (OP)')
to_retain.append('11146 Regenerator Base Temperature Port 8 LHS')
to_retain.append('15070 Regenerator Crown Temperature Port 6 RHS')
to_retain.append('11221 Services Building MCC1 - MV67')
to_retain.append('11217 Services Building MCC9 Cat \'B\' Supply - MV60')
to_retain.append('8344 Total CCCW Flow Measurement')
to_retain.append('136 Total Combustion Air Flow Measurement')
to_retain.append('135 Total Firm Gas Flow Measurement')
to_retain.append('30060 U/S Flowing End Air Flow Measurement')
to_retain.append('11301 UK5 Total Load (Power)')
'''
# ----------------------------------------------------------------------------
# LOAD DATA FOR TRANING AND TESTING
# ----------------------------------------------------------------------------

# Initialise empty data frames
X_df, X_df_test = pd.DataFrame(), pd.DataFrame()
Y_df, Y_df_test = pd.DataFrame(), pd.DataFrame()
Y_raw_df, Y_raw_df_test = pd.DataFrame(), pd.DataFrame()

# Loop over available files of post-processed data
for i in range(1, 5):
    file_name = ('Input Post-Processing ' + str(i) + ' ' +
                 scanner + '.xlsx')

    # The first 3 files are appended to become a single training data-frame
    if i < 4:
        X_df = X_df._append(pd.read_excel(file_name,
                                         sheet_name='input_data'))
        Y_df = Y_df._append(pd.read_excel(file_name,
                                         sheet_name='output_data'))
        Y_raw_df = Y_raw_df._append(pd.read_excel(file_name,
                                   sheet_name='raw_output_data'))

    # The fourth file is used to create the testing data-frame
    if i == 4:
        X_df_test = X_df_test._append(pd.read_excel(file_name,
                                     sheet_name='input_data'))
        Y_df_test = Y_df_test._append(pd.read_excel(file_name,
                                     sheet_name='output_data'))
        Y_raw_df_test = Y_raw_df_test._append(pd.read_excel(file_name,
                                             sheet_name='raw_output_data'))

        # Extract time lags from final file (should be the same for all)
        T_df = pd.read_excel(file_name, sheet_name='time_lags')

# Check data frames are the correct size and have the same column names
assert np.all(X_df.columns == X_df_test.columns)
assert np.all(X_df.columns == T_df.columns)
assert len(X_df) == len(Y_df)
assert len(Y_df) == len(Y_raw_df)
assert len(X_df_test) == len(Y_df_test)
assert len(Y_df_test) == len(Y_raw_df_test)

# ----------------------------------------------------------------------------
# REMOVE INPUTS WE ARE NOT GOING TO USE
# ----------------------------------------------------------------------------

input_names = X_df.columns
for name in input_names:
    if name not in to_retain:
        X_df.drop(columns=name, inplace=True)
        X_df_test.drop(columns=name, inplace=True)
        T_df.drop(columns=name, inplace=True)

# Check that the data frames contain the correct number of inputs
assert len(X_df.columns) == len(to_retain)

# Check that the data frame input names match those in to_retain
assert set(X_df.columns) == set(to_retain)

# ----------------------------------------------------------------------------
# PRE-PROCESSING
# ----------------------------------------------------------------------------

# Process training data
X, Y, N, D, max_lag, time_lags = dpm.align_arrays(X_df, Y_df, T_df)

# standardise data
X_train_stand = minmax.fit_transform(X)
Y_train_stand = minmax.fit_transform(Y)

# Process testing data
X_test, Y_test, N_test, D, max_lag, time_lags = dpm.align_arrays(X_df_test,
                                                                 Y_df_test,
                                                                 T_df)
X_test_stand = minmax.fit_transform(X_test)
Y_test_stand = minmax.fit_transform(Y_test)

# Process raw target data in the same way as the post-processed
# target data. Note this essentially just removes the first max_lag
# points from the date_time array.
Y_raw = dpm.adjust_time_lag(Y_raw_df['raw_furnace_faults'].values,
                            shift=0,
                            to_remove=max_lag)

Y_raw_test = dpm.adjust_time_lag(Y_raw_df_test['raw_furnace_faults'].values,
                                 shift=0,
                                 to_remove=max_lag)

# Extract corresponding time stamps. Note this essentially just
# removes the first max_lag points from the date_time array.
date_time = dpm.adjust_time_lag(Y_df['Time stamp'].values,
                                shift=0,
                                to_remove=max_lag)

date_time_test = dpm.adjust_time_lag(Y_df_test['Time stamp'].values,
                                     shift=0, to_remove=max_lag)