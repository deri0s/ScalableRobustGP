import pandas as pd
import numpy as np
import datetime as dt
from matplotlib import pyplot as plt
import data_processing_methods as dpm

"""
Script for processing input and output data, using methods from
data_process_methods.py

"""

# ----------------------------------------------------------------------------
# USER OPTIONS
# ----------------------------------------------------------------------------

# Choose scanner (either 'MK4' or 'ISRA')
scanner = 'ISRA'

# Choose whether or not to process combinations
process_combinations = True

# Name of spreadsheet specifying tags we are going to keep in, filtering etc.
master_spreadsheet = 'UK5 AI Furnace Model Input Pre-Processing.xlsx'

# Choose from available time periods
dataset_no = 4

if dataset_no == 1:
    initial_date = '21/09/19'
    final_date = '25/10/19'
    generated_file_name = 'Input Post-Processing 1 ' + scanner + '.xlsx'
    tag_folder = 'Tag_Data'
if dataset_no == 2:
    initial_date = '07/03/20'
    final_date = '19/03/20'
    generated_file_name = 'Input Post-Processing 2 ' + scanner + '.xlsx'
    tag_folder = 'Tag_Data'
if dataset_no == 3:
    initial_date = '25/07/20'
    final_date = '03/01/21'
    generated_file_name = 'Input Post-Processing 3 ' + scanner + '.xlsx'
    tag_folder = 'Tag_Data_2020_2021'
if dataset_no == 4:
    initial_date = '19/03/21'
    final_date = '27/06/21'
    generated_file_name = 'Input Post-Processing 4 ' + scanner + '.xlsx'
    tag_folder = 'Tag_Data_2021_03_19_to_2021_06_27'

# ----------------------------------------------------------------------------
# PROCESS FURNACE FAULTS
# ----------------------------------------------------------------------------

# Load furnace fault data frame
ff_df = pd.read_excel('Fault Scanner Data 20190801 to 20210627.xlsx',
                      sheet_name=scanner)

# Extract furnace faults we want to predict and associated time stamps
ff_raw = ff_df['TotalFaults'].values
ff_date_time = pd.to_datetime(ff_df['DateTime'], yearfirst=True)

# Isolate time period we are interested in
ff_raw, ff_date_time = dpm.isolate_time_period(ff_raw, ff_date_time,
                                               initial_date, final_date)

# Process data depending on scanner type
if scanner == 'MK4':
    # Boolean array where 'True' represents location of a stripe in the
    # fault scanner data
    stripe_locations = dpm.identify_stripes(ff_df, scanner,
                                            initial_date, final_date,
                                            plot=False)
    # Remove spikes
    ff = dpm.remove_spikes(ff_raw, tag_name='Furnace faults',
                           tag_ID='', plot=False, olr_def=1)

elif scanner == 'ISRA':

    # initial_date and final_date as NumPy datetime64 to use the process_ISRA
    # method
    str1 = dt.datetime.strptime(initial_date, "%d/%m/%y").strftime("%Y-%m-%d")
    str2 = dt.datetime.strptime(final_date, "%d/%m/%y").strftime("%Y-%m-%d")

    date1 = np.datetime64(dt.datetime.strptime(str1, "%Y-%m-%d"))
    date2 = np.datetime64(dt.datetime.strptime(str2, "%Y-%m-%d"))

    # Process ISRA data by replacing offending module data
    [ff_processed,
     high_ff_locations,
     low_ff_locations] = dpm.process_ISRA(ff_df, date1, date2, plot=False)

    # Remove spikes
    ff = dpm.remove_spikes(ff_processed, tag_name='Furnace faults',
                           tag_ID='', plot=False, olr_def=3)

# Filter data
ff = dpm.low_pass_filter(ff, wn=0.01,
                         tag_name='Furnace faults',
                         tag_ID='', plot=False)

# ----------------------------------------------------------------------------
# PROCESS MODEL INPUTS
# ----------------------------------------------------------------------------

# Load spreadsheet specifying tags we are going to keep in, filtering etc.
meta_data = pd.read_excel(master_spreadsheet, sheet_name='UK5 Tags')

# Initialise list that will store the processed data, name, ID and
# time lags associated with all of the inputs that we end up keeping
# in the model.
X = []
all_ids_plus_names = []
all_time_lags = []

if process_combinations:

    # Isolate all unique indirect tag IDs
    indirect_tags = meta_data['Indirect TagID'].values
    indirect_tags = indirect_tags[~np.isnan(indirect_tags)]
    indirect_tags = np.unique(indirect_tags).astype(int)

    # Loop over signals marked for combination
    print('Processing combination signals...')
    for indirect_tag in indirect_tags:

        # This loop only looks at combustion air flow
        if indirect_tag.astype(str)[0] == '1':

            print('Indirect tag ' + indirect_tag.astype(str))

            # Identify the pairs of signals that need to be combined
            pairs = np.where(meta_data['Indirect TagID'].values ==
                             indirect_tag)

            # Loop over pairs of signals that need to be combined
            for pair in pairs:

                # Create a string of tag ID and name (needed for plots etc.)
                tag_name1 = meta_data['Description'].values[pair[0]]
                tag_name2 = meta_data['Description'].values[pair[1]]
                tag_ID1 = str(int(meta_data['TagID'].values[pair[0]]))
                tag_ID2 = str(int(meta_data['TagID'].values[pair[1]]))

                # Extract time lag (note that both signals should have the same
                # time lag)
                time_lag, load_error = dpm.load_time_lag(meta_data, tag_ID1)

                # If no load error
                if not load_error:

                    # Load raw signals into a data frame
                    df1, load_error = dpm.load_raw_tag_data(tag_ID1, tag_folder)
                    df2, load_error = dpm.load_raw_tag_data(tag_ID2, tag_folder)

                    # If no load error
                    if not load_error:

                        # Extract data values and time stamps
                        x1 = df1['AValue'].values
                        date_time1 = pd.to_datetime(df1['DateTimeStamp'],
                                                    dayfirst=True)
                        x2 = df2['AValue'].values
                        date_time2 = pd.to_datetime(df2['DateTimeStamp'],
                                                    dayfirst=True)

                        # Isolate time period we are interested in
                        x1, date_time1 = dpm.isolate_time_period(x1, date_time1,
                                                                 initial_date,
                                                                 final_date)

                        x2, date_time2 = dpm.isolate_time_period(x2, date_time2,
                                                                 initial_date,
                                                                 final_date)

                        # Convert to total seconds since start of the first
                        # furnace fault measurement we are interested in
                        x1_ts, ff_ts = dpm.convert_to_seconds(date_time1,
                                                              ff_date_time)

                        x2_ts, ff_ts = dpm.convert_to_seconds(date_time2,
                                                              ff_date_time)

                        # The two signals are concatenated together,
                        # then ordered according to their time stamps
                        x_ts_concat = np.concatenate((x1_ts, x2_ts))
                        x_concat = np.concatenate((x1, x2))
                        inds = x_ts_concat.argsort()
                        x_ts_combined = x_ts_concat[inds[::-1]]
                        x_combined = x_concat[inds[::-1]]
                        same_ts_index = []

                        # Take maximum at identical time stamps
                        for i in range(len(x_combined) - 1):
                            if(x_ts_combined[i] == x_ts_combined[i+1]):
                                x_combined[i] = max(x_combined[i], x_combined[i+1])
                                same_ts_index.append(i+1)

                        # Delete data at identical time stamps
                        x_combined = np.delete(x_combined, same_ts_index)
                        x_ts_combined = np.delete(x_ts_combined, same_ts_index)

                        # The name of the combined signal
                        tag_name_combined = ""
                        for i in range(len(tag_name1)):
                            if (tag_name1[i] == tag_name2[i]):
                                tag_name_combined += tag_name1[i]
                            else:
                                break
                        tag_name_combined = tag_name_combined + '(combined)'

                        # Remove spikes
                        x_combined = dpm.remove_spikes(x_combined)

                        # Apply low pass filter
                        x_combined = dpm.low_pass_filter(x_combined, 0.1)

                        # Interpolate over the time stamps where we have furnace
                        # fault measurements
                        x_combined = dpm.interpolate(x_combined,
                                                     x_ts_combined, ff_ts)

                        # Save signals as inputs
                        all_ids_plus_names.append(tag_name_combined)
                        all_time_lags.append(time_lag)
                        X.append(x_combined)

        # This loop only process front wall temperatures
        if indirect_tag.astype(str)[0] == '2':

            plot = False
            if plot:
                fig, ax = plt.subplots()
                ax.set_title('Front wall temperatures averaged', fontsize=15)

            print('Indirect tag ' + indirect_tag.astype(str))

            # Front wall temperature indices
            indices = np.where(meta_data['Indirect TagID'].values == 200000)[0]
            N = len(indices)
            x_sum = 0

            for i in indices:

                # Create a string of tag ID and name (needed for plots etc.)
                tag_name = meta_data['Description'].values[i]
                tag_ID = str(int(meta_data['TagID'].values[i]))

                # Extract time lag
                time_lag, load_error = dpm.load_time_lag(meta_data, tag_ID)

                # If no load error
                if not load_error:

                    # Load raw signals into a data frame
                    df, load_error = dpm.load_raw_tag_data(tag_ID, tag_folder)

                    # If no load error
                    if not load_error:

                        # Extract data values and time stamps
                        x = df['AValue'].values
                        date_time = pd.to_datetime(df['DateTimeStamp'],
                                                   dayfirst=True)

                        # Isolate time period we are interested in
                        x, date_time = dpm.isolate_time_period(x, date_time,
                                                               initial_date,
                                                               final_date)

                        # Convert to total seconds since start of the first
                        # furnace fault measurement we are interested in
                        x_ts, ff_ts = dpm.convert_to_seconds(date_time,
                                                             ff_date_time)

                        # Remove spikes if required
                        if meta_data['Filter Spikes'].values[i] == 1:
                            x = dpm.remove_spikes(x, tag_name, tag_ID, plot=False)

                        # Apply low-pass filter if required
                        if meta_data['Filter Low Pass'].values[i] == 1:
                            x = dpm.low_pass_filter(x, 0.1, tag_name, tag_ID,
                                                    plot=False)

                        # Interpolate over the time stamps where we have furnace
                        # fault measurements
                        x = dpm.interpolate(x, x_ts, ff_ts, tag_name, tag_ID,
                                            plot=False)

                        # Sum up front wall temperatures
                        if i < indices[-1]:
                            x_sum += x

                            if plot:
                                ax.plot(x, color='blue')
                        else:
                            # Calculate the front wall temperatures average
                            x = x_sum/N

            if plot:
                ax.plot(x, color='red', linewidth=4,
                        label='Averaged signal')
                ax.legend()

            # Save the averaged signal
            tag_name = 'Front Wall Temperature Average (PV)'
            all_ids_plus_names.append('200000' + ' ' + tag_name)
            all_time_lags.append(time_lag)
            X.append(x)

        # This loop only looks at regenerator base and crown temperatures
        if indirect_tag.astype(str)[0] == '3':
            print('Indirect tag ' + indirect_tag.astype(str))

            # Identify the pairs of signals that need to be combined
            pairs = np.where(meta_data['Indirect TagID'].values ==
                             indirect_tag)

            # Loop over pairs of signals that need to be combined
            for pair in pairs:

                # Create a string of tag ID and name (needed for plots etc.)
                tag_name1 = meta_data['Description'].values[pair[0]]
                tag_name2 = meta_data['Description'].values[pair[1]]
                tag_ID1 = str(int(meta_data['TagID'].values[pair[0]]))
                tag_ID2 = str(int(meta_data['TagID'].values[pair[1]]))

                # Extract time lag (note that both signals should have the same
                # time lag)
                time_lag, load_error = dpm.load_time_lag(meta_data, tag_ID1)

                # If no load error
                if not load_error:

                    # Load raw signals into a data frame
                    df1, load_error = dpm.load_raw_tag_data(tag_ID1, tag_folder)
                    df2, load_error = dpm.load_raw_tag_data(tag_ID2, tag_folder)

                    # If no load error
                    if not load_error:

                        # Extract data values and time stamps
                        x1 = df1['AValue'].values
                        date_time1 = pd.to_datetime(df1['DateTimeStamp'],
                                                    dayfirst=True)
                        x2 = df2['AValue'].values
                        date_time2 = pd.to_datetime(df2['DateTimeStamp'],
                                                    dayfirst=True)

                        # Isolate time period we are interested in
                        x1, date_time1 = dpm.isolate_time_period(x1, date_time1,
                                                                 initial_date,
                                                                 final_date)

                        x2, date_time2 = dpm.isolate_time_period(x2, date_time2,
                                                                 initial_date,
                                                                 final_date)

                        # Convert to total seconds since start of the first
                        # furnace fault measurement we are interested in
                        x1_ts, ff_ts = dpm.convert_to_seconds(date_time1,
                                                              ff_date_time)

                        x2_ts, ff_ts = dpm.convert_to_seconds(date_time2,
                                                              ff_date_time)

                        # Apply reversals
                        x1, x1_ts = dpm.reversal(x1, x1_ts, tag_name1, tag_ID1,
                                                 plot=False)
                        x2, x2_ts = dpm.reversal(x2, x2_ts, tag_name2, tag_ID2,
                                                 plot=False)

                        # Remove spikes if required
                        if meta_data['Filter Spikes'].values[pair[0]] == 1:
                            x1 = dpm.remove_spikes(x1, tag_name1, tag_ID1,
                                                   plot=False)
                        if meta_data['Filter Spikes'].values[pair[1]] == 1:
                            x2 = dpm.remove_spikes(x2, tag_name2, tag_ID2,
                                                   plot=False)

                        # Apply low pass filter if required
                        if meta_data['Filter Low Pass'].values[pair[0]] == 1:
                            x1 = dpm.low_pass_filter(x1, 0.1, tag_name1, tag_ID1,
                                                     plot=False)
                        if meta_data['Filter Low Pass'].values[pair[1]] == 1:
                            x2 = dpm.low_pass_filter(x2, 0.1, tag_name2, tag_ID2,
                                                     plot=False)

                        # Interpolate over the time stamps where we have furnace
                        # fault measurements
                        x1 = dpm.interpolate(x1, x1_ts, ff_ts, tag_name1, tag_ID1,
                                             plot=False)

                        x2 = dpm.interpolate(x2, x2_ts, ff_ts, tag_name2, tag_ID2,
                                             plot=False)

                        # Calculate absolute difference between the signals
                        x_diff = np.abs(x2 - x1)

                        # The name of the combined signal
                        tag_name_combined = ""
                        for i in range(len(tag_name1)):
                            if (tag_name1[i] == tag_name2[i]):
                                tag_name_combined += tag_name1[i]
                            else:
                                break
                        tag_name_combined = tag_name_combined + '(abs. difference)'

                        # Save the average signal
                        all_ids_plus_names.append(tag_name_combined)
                        all_time_lags.append(time_lag)
                        X.append(x_diff)

# Loop over all non-combination inputs in meta_data_df
for i in range(len(meta_data['TagID'].values)):
    print('i = ', i)

    # Only continue if input is enabled
    if meta_data['Enable'].values[i] == 1:

        # In this loop we don't process any combination signals (they
        # are treated separately). If a signal is not needed for a
        # combination that the 'Indirect TagID' will be NaN.
        if np.isnan(meta_data['Indirect TagID'].values[i]):

            # Create a string of tag ID and name (needed for plots etc.)
            tag_name = meta_data['Description'].values[i]
            tag_ID = str(int(meta_data['TagID'].values[i]))
            print('\t' + tag_ID + ' ' + tag_name)

            # Extract time lag
            time_lag, load_error = dpm.load_time_lag(meta_data, tag_ID)

            # If no load error
            if not load_error:

                # Load raw signal into a data frame
                df, load_error = dpm.load_raw_tag_data(tag_ID, tag_folder)

                # If no load error
                if not load_error:

                    # Extract data values and time stamps
                    x = df['AValue'].values
                    date_time = pd.to_datetime(df['DateTimeStamp'],
                                               dayfirst=True)

                    # Isolate time period we are interested in
                    x, date_time = dpm.isolate_time_period(x, date_time,
                                                           initial_date,
                                                           final_date)

                    # Convert to total seconds since start of the first
                    # furnace fault measurement we are interested in
                    x_ts, ff_ts = dpm.convert_to_seconds(date_time,
                                                         ff_date_time)

                    # Test to see if it falls outside of engineering ranges and
                    # process if required
                    x, min_max_pass = dpm.eng_min_max(x,
                                                      meta_data['Eng Min'][i],
                                                      meta_data['Eng Max'][i],
                                                      tag_name, tag_ID,
                                                      plot=False)
                    if min_max_pass:
                        print('\t Outside of min-max range')

                    # Remove spikes if required
                    if meta_data['Filter Spikes'].values[i] == 1:
                        x = dpm.remove_spikes(x, tag_name, tag_ID, plot=False)
                        print('\t Removing spikes')

                    # Apply 'reversals' conditioning if required
                    if meta_data['Filter Reversal'].values[i] == 1:
                        x, x_ts = dpm.reversal(x, x_ts, tag_name, tag_ID,
                                               plot=False)
                        print('\t Applying reversals conditioning')

                    # Low pass filter
                    if meta_data['Filter Low Pass'].values[i] == 1:
                        wn = meta_data['Parameter Low Pass'].values[i]
                        x = dpm.low_pass_filter(x, wn, tag_name, tag_ID,
                                                plot=False, order=3)
                        print('\t Applying low pass filter')

                    # Interpolate over the time stamps where we have furnace
                    # fault measurements
                    x = dpm.interpolate(x, x_ts, ff_ts, tag_name, tag_ID,
                                        plot=False)

                    # If tag data has made it this far then we are
                    # going to keep it as an input so we append it
                    all_ids_plus_names.append(tag_ID + ' ' + tag_name)
                    all_time_lags.append(time_lag)
                    X.append(x)

# ----------------------------------------------------------------------------
# SAVE POST-PROCESSED DATA AS SPREADSHEET
# ----------------------------------------------------------------------------

X_d = {}      # Initialise dictionary for inputs
Y_d = {}      # Initialise dictionary for outputs
Yraw_d = {}   # Initialise dictionary for raw (pre-processed) outputs
T_d = {}      # Initialise dictionary for time lags
M_d = {}      # Initialise dictionary for any meta data

# Store initial and final date of time period we're interested in as
# meta data
M_d['Initial date'], M_d['Final date'] = initial_date, final_date

# Create dictionary of time lags
for i in range(len(all_time_lags)):
    T_d[all_ids_plus_names[i]] = all_time_lags[i]

# Create dictionary of inputs
for i in range(len(all_ids_plus_names)):
    X_d[all_ids_plus_names[i]] = X[i]

# Create dictionary of outputs
Y_d['furnace_faults'] = ff
Y_d['Time stamp'] = ff_date_time
if scanner == 'MK4':
    Y_d['Stripe'] = stripe_locations
elif scanner == 'ISRA':
    Y_d['High values replaced'] = high_ff_locations
    Y_d['Low values replaced'] = low_ff_locations

# Create dictionary of pre-processed outputs
Yraw_d['raw_furnace_faults'] = ff_raw

# Create data frames of final processed data and save as excel file
X_df = pd.DataFrame(X_d)
Y_df = pd.DataFrame(Y_d)
Yraw_df = pd.DataFrame(Yraw_d)
T_df = pd.DataFrame(T_d, index=np.array([1]))
M_df = pd.DataFrame(M_d, dtype=str, index=np.array([1]))

# Define an Excel writer object and the target file
writer = pd.ExcelWriter(generated_file_name)

# Save to spreadsheet
X_df.to_excel(writer, sheet_name='input_data', index=False)
Y_df.to_excel(writer, sheet_name='output_data', index=False)
Yraw_df.to_excel(writer, sheet_name='raw_output_data', index=False)
T_df.to_excel(writer, sheet_name='time_lags', index=False)
M_df.to_excel(writer, sheet_name='meta_data', index=False)
writer.save()

# If there are any plots to show
plt.show()
