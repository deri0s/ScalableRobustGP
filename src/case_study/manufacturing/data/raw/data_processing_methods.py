import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

def process_ISRA(df, date1, date2, plot=False):

    """
    Description
    -----------
        Process ISRA scanner data by replacing high module data with the
        average of remaining modules and low ff values with the previous
        reading and plots (if requested).

    Parameters
    ----------
        df : pandas data frame of fault scanner data.

        date1 : first date in the range we are interested in.

        date2 : second date in the range we are interested in.

        plot : boolean option to create plot

    Returns
    -------

        total_faults_processed : the total fault density after replacing module data

        to_replace_high : Identified locations of high ff values

        to_replace_low: Identified locations of low ff values

    """

    # Extract scanner data
    X, N, date_time, total_faults = extract_ISRA_modules(df, date1, date2)

    # Classify
    to_replace_high= np.repeat(False, N)
    to_replace_low= np.repeat(False, N)
    for i in range(N):
        if total_faults[i] > 5:
            to_replace_high[i] = True
        if total_faults[i] < 0.05:
            to_replace_low[i] = True

    # Find the module with maximum fault density
    max_module = np.argmax(X[:,to_replace_high],axis=0)
    X_processed = np.copy(X)
    X_processed[max_module,to_replace_high] = 0

    # Find the average of remaining modules and replace the offending module
    X_processed[max_module,to_replace_high] = np.sum(X_processed[:,to_replace_high],0)/(np.size(X,0)-1)
    total_faults_processed = np.sum(X_processed, 0)

    for i in range(N):
        if to_replace_low[i]:
            total_faults_processed[i] = total_faults_processed[i-1]

    if plot:

        # Initialise
        fig, ax = plt.subplots(nrows=2)
        fig.autofmt_xdate()

        # Plot total faults after data replacement
        ax[0].plot(date_time, total_faults, color='black')
        ax[0].plot(date_time[to_replace_high], total_faults[to_replace_high], 'o',
                   color='red', label='To replace (High)')
        ax[0].plot(date_time[to_replace_low], total_faults[to_replace_low], 'o',
                   color='orange', label='To replace (Low)')
        ax[0].plot(date_time, total_faults_processed, color='green', label='processed')
        ax[0].set_ylabel('Total fault density')
        ax[0].set_xlim([date1, date2])
        ax[0].legend()

        # Use imshow to show individual module values
        xlims = mdates.date2num([date1, date2])
        ylims = [4.5, -0.6]
        lims = [xlims[0], xlims[1], ylims[0], ylims[1]]
        im = ax[1].imshow(X_processed, aspect='auto', extent=lims, vmin=0, vmax=20)
        ax[1].xaxis_date()
        ax[1].set_yticks(np.arange(0, 5))
        ax[1].set_ylim(ylims)
        ax[1].set_ylabel('Module no.')

    return total_faults_processed, to_replace_high, to_replace_low


def extract_ISRA_modules(df, date1, date2, plot=False):
    """
    Description
    -----------
        Loads data from the individual modules for the ISRA
            scanner and plots (if requested).

    Parameters
    ----------
        df : pandas data frame of fault scanner data.

        date1 : first date in the range we are interested in.

        date2 : second date in the range we are interested in.

        plot : boolean option to create plot

    Returns
    -------
        X : 5xN array of individual module values

        N : total no. of samples over time period

        date_time : date time array associated with samples

        total_faults : the total fault density per time stamp

    """

    # Boolean array of indices where data range occurs
    indx = np.logical_and(df['DateTime'].values > date1,
                          df['DateTime'].values < date2)

    # DateTime array only over the user-specified date range
    date_time = df['DateTime'].values[indx]

    # Initialise matrix to contain module values
    N = np.sum(indx)
    X = np.zeros([5, N])

    # Extract readings per module
    for i in range(5):
        name = 'Module' + str(i) + 'Faults'
        X[i, :] = df[name].values[indx]

    # Calculate the total no. of faults
    total_faults = np.sum(X, 0)

    # Plot option
    if plot:

        # Initialise
        fig, ax = plt.subplots(nrows=2)
        fig.autofmt_xdate()

        # Plot total faults
        ax[0].plot(date_time, total_faults, color='black')
        ax[0].set_ylabel('Total fault density')
        ax[0].set_xlim([date1, date2])
        ax[0].legend()

        # Use imshow to show individual module values
        xlims = mdates.date2num([date1, date2])
        ylims = [4.5, -0.6]
        lims = [xlims[0], xlims[1], ylims[0], ylims[1]]
        im = ax[1].imshow(X, aspect='auto', extent=lims)
        ax[1].xaxis_date()
        ax[1].set_yticks(np.arange(0, 5))
        ax[1].set_ylim(ylims)
        ax[1].set_ylabel('Module no.')

    return X, N, date_time, total_faults


def entropy(x):
    """
    Description
    -----------
        Estimate the Shannon entropy of a distribution from a histogram
            of samples. Note that -p(x)log(p(x)) -> 0 as p(x)-> 0 so
            bins containing zero samples will be assigned entropy equal to
            zero.

    Parameters
    ----------
        x : numpy array of samples

    Returns
    -------
        H : estiamted Shannon entropy

    """

    # Normalise histogram to realise estimate of the probability distribution
    # over x.
    px = x / np.sum(x)

    # Estimate the Shannon entropy associated with each bin in the histogram
    h = np.zeros(len(x))
    for i in range(len(x)):
        if px[i] > 0:
            h[i] = -px[i] * np.log(px[i])

    # Estimate the total Shannon entropy for the distribution
    H = np.sum(h)
    return H


def identify_stripes(ff_df, scanner, initial_date, final_date, plot=False):
    """
    Description
    -----------
        Identifies 'stripes' in the fault density signal which, we think,
            are not related to furnace faults. For the MK4 scanner the
            stripes are across the ribbon (vertical in the following plots)
            but for the ISRA 5D scanner the stripes are along the ribbon
            (horizontal) in the following plots.

        In the following, stripes are identified as those signals which
            have high estimated Shannon entropy while also contributing
            highly to the overall estimated fault density measurement.

        Note that, throughout, we adopt the convention that an outcome with
            estimated probability equal to 0 does not contribute to the
            estimated Shannon entropy (something you can prove with l'Hopital's
            rule).

        NOTE: we currently do not have an approach for the ISRA5D scanner.

    Parameters
    ----------
        ff_df : furnace fault data frame (including readings for each module)

        scanner : name of the scanner (either 'MK4' or 'ISRA')

        initial_date : string of first date in the time period we are
            interested in

        final_date : string of final date in the time period we are
            interested in

        plot : boolean plot option

    Returns
    -------
        stripe_loc : vector of boolean values, where 'True' represents a
            spike location

    """

    # Date range we're interested in
    initial_date_time = pd.to_datetime(initial_date, dayfirst=True)
    final_date_time = pd.to_datetime(final_date, dayfirst=True)

    # Boolean array showing indices where date range occurs
    date_time = pd.to_datetime(ff_df['DateTime'], yearfirst=True)
    i1 = date_time > initial_date_time
    i2 = date_time < final_date_time
    indx = i1.values * i2.values

    # Total no. of points to look at
    N = np.sum(indx)

    if scanner == 'MK4':

        # Create 'X'; an array whose columns are the measurements associated
        # with each module in the MK4 scanner
        module_names = ['Module0Faults', 'Module1Faults', 'Module2Faults',
                        'Module3Faults', 'Module4Faults', 'Module5Faults',
                        'Module6Faults', 'Module7Faults', 'Module8Faults',
                        'Module9Faults', 'Module10Faults', 'Module11Faults',
                        'Module12Faults', 'Module13Faults', 'Module14Faults',
                        'Module15Faults']
        X = ff_df[module_names].values[indx]

        # Check that X is the correct size
        assert np.shape(X)[0] == N
        assert np.shape(X)[1] == 16

        # Calculate entropy values
        entropy = np.zeros(N)
        for i in range(N):

            # Ensure px sums to one
            px = X[i, :] / np.sum(X[i, :])

            # Calculate estimated entropy (using the convention that
            # -px log(px) => 0 as px => 0
            indx_zeros = np.where(px == 0)
            log_px = np.log(px)
            log_px[indx_zeros] = 0
            entropy[i] = np.sum(-px * log_px)

        # Find summation of faults reported in each module
        total_faults = np.sum(X, 1)

        # Find stripes
        stripe_loc = np.logical_and(total_faults > 2.5, entropy > 2)

        if plot:

            # Plot entropy vs. total faults
            fig, ax = plt.subplots()
            ax.plot(entropy, total_faults, 'o', color='black',
                    label='No stripe')
            ax.plot(entropy[stripe_loc], total_faults[stripe_loc],
                    'o', color='red', label='Stripe')
            ax.set_xlabel('Entropy')
            ax.set_ylabel('Total Faults')
            ax.legend()

            # Plot image of ribbon (removing very high individual module
            # measurements to help visualisation)
            fig, ax = plt.subplots()
            indx_high = X > 3
            X[indx_high] = 0
            im = plt.imshow(X.T, aspect='auto')
            fig.colorbar(im)
            ax.set_xlim([0, N])
            ax.set_ylim([-1, 15])
            ax.set_yticks(np.arange(0, 16))
            ax.set_yticklabels(module_names)
            ax.set_title('Measurements per module (note: capped at 3)')
            ax.plot(np.arange(0, N)[stripe_loc],
                    np.repeat(-0.5, np.sum(stripe_loc)), 'o', color='red')
            plt.show()

    elif scanner == 'ISRA':
        raise SyntaxError('\n No stripes approach for the ISRA 5D \n')
    else:
        raise SyntaxError('\n Scanner name not recognised \n')

    return stripe_loc


def reversal(x, x_ts, tag_name=None, tag_ID=None, plot=False):
    """
    Description
    -----------
        Get the general trend of a signal affected by the reversal process.
        The algorithm finds the max value within a 40 min time range.

    Parameters
    ----------
    x : numpy array of the signal affected by the reversal process.
    x_ts : pandas series of the total seconds associated with the signal x.
    tag_name : String of the tag name. Only needed if we are going to create
               a plot.
    tag_ID : String of tag ID. Only needed if we are going to create a plot.
    plot : boolean value stating whether or not to plot the result.

    Returns
    -------
    retained_ts : pandas series of the total seconds retained after the
                  signal rectification
    retained_x : Returns the general trend of the signal
    """

    # Make sure x is a numpy array
    x = np.array(x)

    # The index at the first firing
    indices = np.where(x_ts.values - x_ts.values[0] < 2400)[0]
    indx_first_firing = np.argmin(x[indices])

    # Outer while loop counter:
    i = 0       # The number of elements required to get up to 40 minutes

    # Inner while loop counter:
    j = 0       # The number of 40 minute time windows in the x_ts vector

    # The starting point for the first 40 minute time window
    start = np.copy(indx_first_firing)

    # Initialise static vector to save memory
    max_pos = np.zeros(len(x))

    # ------------------------------------------------------------------------
    # GET THE MAX VALUES AT EACH 40 MINUTE TIME WINDOW
    # ------------------------------------------------------------------------

    while start < len(x):
        seconds = 0
        i = 0

        # Inner loop: Collect time-stamp elements up to 40 minutes
        while seconds < 2400 and start + i < len(x):
            seconds = int(x_ts.iloc[start + i] - x_ts.iloc[start])
            i += 1

        # The 40 minute time window
        window = int(start + i)
        max_pos[j] = int(start + np.argmax(x[start:window]))

        # Start at the next 40 minutes time range
        start = start + i
        j += 1

    # Covert to list of integers checking for zero values
    retained_pos = [int(i) for i in max_pos if i != 0.0]

    # The rectified signal and its corresponding time-stamps
    retained_ts = x_ts.iloc[retained_pos]
    retained_x = x[retained_pos]

    if plot:
        indx = np.arange(len(x))
        plt.figure()
        title = tag_ID + ' ' + tag_name
        plt.title('Reversals processing: ' + '\n'+title, fontsize=20)
        plt.plot(indx, x, color='gray', label='with reversals')
        plt.plot(retained_pos, retained_x, color='red', label='rectified')
        plt.axvline(indx_first_firing, linewidth=3, color='green',
                    label='starting point')
        plt.xlabel('Index', fontsize=13)
        plt.legend(prop={'size': 13})

    return retained_x, retained_ts


def load_time_lag(meta_data, tag_ID):
    """
    Description
    -----------
        Loads the time lag associated with a particular tag id

    Parameters
    ----------
        meta_data : pandas data frame of the main pre-processing file
        tag_ID : str value of the particular tag id that we're interested in

    Returns
    -------
        time_lag : the time lag associated with the tag that has
                   id tag_ID
    """

    tag_ID = int(float(tag_ID))

    # Try to load appropriate tag, report error if we can't find it
    try:
        i = meta_data.index[meta_data['TagID'] == tag_ID].tolist()[0]
        load_error = False

        # Time lag in hours
        time_lag_hours = meta_data['Time Lag (Hours)'].values[i]

        # Convert hours to time units. Each unit of time represents 20 min
        time_lag = round(3 * time_lag_hours)

    except ValueError:
        print('Could not find time lag associated with tag_ID',
              str(tag_ID))
        time_lag = []
        load_error = True

    return time_lag, load_error


def load_raw_tag_data(tag_ID, folder_name):
    """
    Description
    -----------
        For a specified tag id number, this method will search through
            the Tag_Data folder of raw data and load the appropriate .txt
            file.

    Parameters
    ----------
        tag_ID : str of the tag id number that we want to load.
        folder_name : in the case where the data is in another folder.

    Returns
    -------
        df : pandas data frame of raw data corresponding to tag_ID
        load_error : is true if the file couldn't be located
    """

    import glob

    # Create a list of all the file names in the Tag_Data folder
    file_names = glob.glob(folder_name+'/*.txt')

    # Loop over each file name and separate out the tag ID from
    # the rest of each name and add to the list all_tag_IDs.
    all_tag_IDs = []
    for names in file_names:
        space1 = names.find(' ', 0)
        space2 = names.find(' ', space1+1)
        all_tag_IDs.append(names[space1+1:space2])

    # Find the index of the txt file that we need to load, then
    # load data into a data frame
    try:
        i = all_tag_IDs.index(str(int(tag_ID)))
        df = pd.read_csv(file_names[i], sep=';')
        load_error = False
    except ValueError:
        print('Could not find tag_ID', str(int(tag_ID)))
        df = []
        load_error = True

    return df, load_error


def isolate_time_period(x, date_time, initial_date, final_date):
    """
    Description
    -----------
        Takes an array of of data and returns a pandas date-time
            and signal values over specified date range.

    Parameters
    ----------
        x : data to be processed
        data_time : a pandas date time object, describing the time
            stamps associated with x
        initial_date : string, first data in the time period that we're
            interested in, in the format 'dd/mm/yy'
        final_date : string, final data in the time period that we're
            interested in, in the format 'dd/mm/yy'

    Returns
    -------
        x_new : signal amplitude over the specified date range
        date_time_new : pandas date time over the specified date range
    """

    # Check that initial_date and final_date are strings
    assert isinstance(initial_date, str)
    assert isinstance(final_date, str)

    # Date range we're interested in
    initial_date_time = pd.to_datetime(initial_date, dayfirst=True)
    final_date_time = pd.to_datetime(final_date, dayfirst=True)

    # Boolean array showing indices where date range occurs
    i1 = date_time > initial_date_time
    i2 = date_time < final_date_time
    i_range = i1.values * i2.values

    # Outputs
    date_time_new = date_time[i_range]
    x_new = x[i_range]

    return x_new, date_time_new


def eng_min_max(x, x_min, x_max, tag_name=None, tag_ID=None, plot=False):
    """
    Description
    -----------
        Tests whether the signal x is within the range [min, max]. If the
        signal is found to be outside of the range, it is set equal to the
        first previous value that was within range. Can plot the pre- and
        post-processed signal and range if we want to.

    Parameters
    ----------
        x : signal to be tested

        min : minimum allowable value

        max : maximum allowable value

        tag_name : String of the tag name. Is only needed if we are
            going to create a plot.

        tag_ID : String of tag ID. Is only needed if we are going to
            create a plot.

        plot : boolean value stating whether or not we will plot the result

    Returns
    -------
        x_processed : processed signal

        min_max_pass : True if passed the test, false otherwise
    """

    # Check to see if the signal has fallen outside of eng min
    # eng max range. Plot if it has.
    less_than_min = x < x_min
    more_than_max = x > x_max

    if np.sum(less_than_min) or np.sum(more_than_max):

        # If doesn't pass, find the points where the signal goes
        # outside of its range; if it does, set it equal to the
        # previous value.
        x_processed = np.copy(x)
        for i in range(len(x)):
            if (x_processed[i] < x_min or x_processed[i] > x_max):
                x_processed[i] = x_processed[i-1]

        # Return boolean value indicating that the original signal
        # was outside of the engineering limits.
        min_max_pass = False

        # Option to plot
        if plot:
            fig, ax = plt.subplots()
            ax.plot(x, color='black', label='Pre-processed')
            ax.plot(x_processed, color='blue', label='Post-processed')
            ax.plot(np.array([0, len(x)]),
                    np.repeat(x_min, 2),
                    color='red')
            ax.plot(np.array([0, len(x)]),
                    np.repeat(x_max, 2),
                    color='red')
            title = (tag_ID + ' ' + tag_name + '\n' + '(outside min-max)')
            ax.set_title(title)
            ax.legend()

    else:
        x_processed = np.copy(x)
        min_max_pass = True

    return x_processed, min_max_pass


def low_pass_filter(x, wn=0.1, tag_name=None, tag_ID=None,
                    plot=False, order=3):
    """
    Description
    -----------
        Passes data through low-pass filter, using filtfilt so that no
            phase difference is induced.

    Parameters
    ----------
        x : signal to be filtered
        tag_name : String of the tag name. Is only needed if we are
            going to create a plot.
        tag_ID : String of tag ID. Is only needed if we are going to
            create a plot.
        plot : boolean value stating whether or not we will plot the result
        order : order of  the filter

    Returns
    -------
        x_filt : the filtered signal
    """

    from scipy.signal import butter, filtfilt

    b, a = butter(N=order, Wn=wn)
    x_filt = filtfilt(b, a, x)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(x, color='red', label='Unfiltered')
        ax.plot(x_filt, color='black', label='Filtered')
        title = tag_ID + ' ' + tag_name + '\n' + '(filtering)'
        ax.set_title(title)
        ax.legend()

    return x_filt


def high_pass_filter(x, tag_name=None, tag_ID=None, plot=False):
    """
    Description
    -----------
        Passes data through high-pass filter, using filtfilt so that no
            phase difference is induced.

    Parameters
    ----------
        x : signal to be filtered
        tag_name : String of the tag name. Is only needed if we are
            going to create a plot.
        tag_ID : String of tag ID. Is only needed if we are going to
            create a plot.
        plot : boolean value stating whether or not we will plot the result

    Returns
    -------
        x_filt : the filtered signal
    """

    from scipy.signal import butter, filtfilt

    b, a = butter(N=3, Wn=0.01, btype='highpass')
    x_filt = filtfilt(b, a, x)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(x, color='red', label='Unfiltered')
        ax.plot(x_filt, color='black', label='Filtered')
        title = tag_ID + ' ' + tag_name + '\n' + '(high-pass filtering)'
        ax.set_title(title)
        ax.legend()

    return x_filt


def convert_to_seconds(date_time, ff_date_time):
    """
    Description
    -----------
        Converts the date-time associated with x into the
            number of seconds since the start of the furnace
            fault data that we are interested in. Also converts
            the furnace fault data into seconds.

    Parameters
    ----------
        date_time : pandas date time associated with x
        ff_date_time : pandas date time associated with
            furnace faults

    Returns
    -------
        x_ts : total seconds associated with signal x
        ff_ts : total seconds associated with furnace faults
    """

    # Time delta between date_time and the start of the date_time associated
    # with the furnace faults.
    x_ts = (pd.to_timedelta(date_time - ff_date_time.iloc[0]).
            dt.total_seconds())
    ff_ts = (pd.to_timedelta(ff_date_time - ff_date_time.iloc[0]).
             dt.total_seconds())

    return x_ts, ff_ts


def interpolate(x, x_ts, ff_ts, tag_name=None, tag_ID=None, plot=False):
    """
    Description
    -----------
        Interpolates to get estimates of x at the furnace fault
            time stamps. Interpolation is conducted using the
            previous value.

    Parameters
    ----------
        x : signal input
        x_ts : total seconds associated with signal x
        ff_ts : total seconds associated with furnace fault signal
        tag_name : String of the tag name. Is only needed if we are
            going to create a plot.
        tag_ID : String of tag ID. Is only needed if we are going to
            create a plot.
        plot : boolean value stating whether or not we will plot the result

    Returns
    -------
        x_int : interpolated x values
    """

    from scipy.interpolate import interp1d

    f = interp1d(x_ts, x, kind='previous', fill_value='extrapolate')
    x_int = f(ff_ts)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(x_ts, x, color='black', marker='.', label='Original')
        ax.plot(ff_ts, x_int, color='red', marker='.',
                label='Interpolated')
        title = tag_ID + ' ' + tag_name + '\n' + '(interpolation)'
        ax.set_title(title)
        ax.legend()

    return x_int


def standardise(x, offset, width):
    """
    Description
    -----------
        Standardise the array, x, so that it removes 'offset' then divides
            by 'width'.

    Parameters
    ----------
        x : signal to be standardised
        offset : fairly obvious...
        width : also fairly obvious...

    Returns
    -------
        xs : standardised signal
    """

    if width == 0:
        print('Warning: standard deviation equal to zero')
        xs = x - offset
    else:
        xs = (x - offset) / width

    return xs


def adjust_time_lag(x, shift, to_remove):
    """
    Description
    -----------
        Shift x using numpy's roll method before removing the first
            to_remove data points. This method is needed to align
            time-lagged inputs with corresponding outputs.

    Parameters
    ----------
        x : signal to be processed
        shift : amount signal will be shifted
        to_remove : no. of points to be removed from start of signal
            after shift has been applied

    Returns
    -------
        x_out : processed version of the signal
    """

    x_out = np.roll(x, shift)
    x_out = x_out[to_remove:]

    return x_out


def remove_spikes(x, tag_name=None, tag_ID=None, plot=False, olr_def=3):
    """
    Description
    -----------
        Removes outlier spikes from the data x and replaces resulting
            missing values using linear interpolation. We pass
            the signal through a high-pass filter before it gets to
            this point to remove slowly varying trends.

    Parameters
    ----------
        x : signal to be processed
        tag_name : String of the tag name. Is only needed if we are
            going to create a plot.
        tag_ID : String of tag ID. Is only needed if we are going to
            create a plot.
        plot : option to plot results of outlier removal
        olr_der : no. of standard deviations beyond the mean that will
            be defined as an outlier

    Returns
    -------
        x_final : x with spikes removed and replaced with data from
            linear interpolation
    """

    x_filt = high_pass_filter(x, tag_name, tag_ID, False)

    m = np.mean(x_filt)
    xn = x_filt - m
    s = np.std(xn)

    # Here we define outlier as being more than 3 standard
    # deviations from the mean
    spike_locations = np.where(np.abs(xn) > olr_def * s)[0]
    indx = np.arange(0, len(x))
    indx_no_spikes = np.delete(indx, spike_locations)

    x_no_spikes = np.delete(x, spike_locations)
    x_final = np.interp(indx, indx_no_spikes, x_no_spikes)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(indx, x, color='black', label='Original signal')
        ax.plot(spike_locations, x[spike_locations], 'o',
                color='red', label='Outliers')
        ax.plot(indx, x_final, color='blue', label='Remaining signal')
        ax.legend()
        title = tag_ID + ' ' + tag_name + '\n' + '(spike removal)'
        ax.set_title(title)

    return x_final


def ff_remove_spikes(x, window_size=9, plot=False):
    """
    Description
    -----------
        Removes outlier spikes from the data x and replaces resulting
            missing values using linear interpolation. We pass
            the signal through a high-pass filter before it gets to
            this point to remove slowly varying trends.

        NOTE: NOT CURRENTLY USED BUT MAY BE USEFUL IN THE FUTURE.

    Parameters
    ----------
        x : signal to be processed
        window_size : size of the time window in 20min time steps
        plot : option to plot results of outlier removal

    Returns
    -------
        x_final : x with spikes removed and replaced with data from
            linear interpolation
    """

    tag_name = 'Furnace faults'
    tag_ID = ''
    x_filt = high_pass_filter(x, tag_name, tag_ID, plot)

    # Calculating mean and standard deviation sequentially using a time window
    # where here we define outliers as being more than 3 sigma from the mean

    i = window_size + 1
    spike_locations = []

    while i < len(x_filt) - window_size:

        past_values = x_filt[i-window_size-1: i-1]
        window_average = np.mean(past_values)
        xn = past_values - window_average
        s = np.std(xn)

        if (np.abs(x_filt[i] - window_average) > 3*s):

            # Looking at future furnace fault values to verify current
            # reading is a spike not the start of an extended period
            # of high fault density

            for j in range(1, window_size):
                if (np.abs(x_filt[i+j] - window_average) < 3*s):
                    spike_locations.append(i)
        i += 1

    indx = np.arange(0, len(x))
    indx_no_spikes = np.delete(indx, spike_locations)

    x_no_spikes = np.delete(x, spike_locations)
    x_final = np.interp(indx, indx_no_spikes, x_no_spikes)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(indx, x, color='black', label='Original signal')
        ax.plot(spike_locations, x[spike_locations], 'o',
                color='red', label='Outliers')
        ax.plot(indx, x_final, color='blue', label='Remaining signal')
        ax.legend()
        title = 'Furnace faults \n (spike removal)'
        ax.set_title(title)

    return x_final


def align_arrays(X_df, Y_df, T_df):
    """
    Description
    -----------
    Takes post-processed data frames as inputs and aligns arrays
        based on estimated time lags, using the adjust_time_lag
        method. Should be the final step before delivering matrix of
        inputs and array of targets. This is explained graphically
        in a powerpoint presentation of the same name.

    Parameters
    ----------
    X_df : post-processed data frame of tag data
    Y_df : post-processed data frame of furnace faults
    T_df : post-processed data frame of time lags

    Returns
    -------
    X : 2D array of inputs (ready for model training)
    Y : array of targets (ready for model training)
    N : no. input-output pairs
    D : dimension of the model input
    max_lag : the maximum identified time lag
    time_lags : array of all time lags
    """

    # Extract time lags
    time_lags = T_df.values[0]

    # Find maximum time lag
    max_lag = np.max(time_lags)

    # Create array of targets
    Y = adjust_time_lag(Y_df['furnace_faults'].values,
                        shift=0,
                        to_remove=max_lag)

    # No. points available for training etc.
    N = len(Y)

    # No. tag inputs (not including auto-regressive input)
    D = np.shape(X_df)[1]

    # Create tag inputs
    X = np.zeros([N, D])
    for i in range(D):

        # Name of input to be processed
        col_name = X_df.columns[i]

        # Use input name to extract signal and time lag
        x = X_df[col_name].values

        # Process signal and add to matrix of inputs
        x = adjust_time_lag(x, shift=time_lags[i],
                            to_remove=max_lag)
        X[:, i] = x

    return X, Y, N, D, max_lag, time_lags


def align_timelags(X_df, Y_df, T_df):
    """
    Description
    -----------
    Only used when estimating the timelags

    Parameters
    ----------
    X_df : post-processed data frame of tag data
    Y_df : post-processed data frame of furnace faults
    T_df : post-processed array of time lags

    Returns
    -------
    X : 2D array of inputs (ready for model training)
    Y : array of targets (ready for model training)
    N : no. input-output pairs
    D : dimension of the model input
    max_lag : the maximum identified time lag
    time_lags : array of all time lags
    """

    # Extract time lags
    time_lags = T_df.values

    # Find maximum time lag
    max_lag = int(np.max(time_lags))

    # Create array of targets
    Y = adjust_time_lag(Y_df['furnace_faults'].values,
                        shift=0,
                        to_remove=max_lag)

    # No. points available for training etc.
    N = len(Y)

    # No. tag inputs (not including auto-regressive input)
    D = np.shape(X_df)[1]

    # Create tag inputs
    X = np.zeros([N, D])
    for i in range(D):

        # Name of input to be processed
        col_name = X_df.columns[i]

        # Use input name to extract signal and time lag
        x = X_df[col_name].values

        # Process signal and add to matrix of inputs
        x = adjust_time_lag(x, shift=int(time_lags[i]), to_remove=max_lag)
        X[:, i] = x

    return X, Y, N, D, max_lag, time_lags


def random_time_lags(time_lags, Ns, s):
    """
    Description
    -----------
        Generates Ns samples of the time lags. Each samples is drawn from
            a Gaussian, centred on the estimated time lag provided by
            Pilkington and with standard deviation s. Note that we are
            using the same standard deviation for each input.

        Any sampled time lags that are less than 0 are set equal to 0.

    Parameters
    ----------
        time_lags - array of the time lag values
        Ns - no. samples to generate.
        s - standard deviation of samples around each time lag

    Returns
    -------
        samples - random samples of time lags
    """

    # Use values in spreadsheet as mean
    mean_time_lags = time_lags

    # Dimension of samples
    D = len(mean_time_lags)

    # Generate samples and round to closest integer
    samples = np.round(mean_time_lags + s * np.random.randn(Ns, D))

    # Set negative values equal to zero
    rows, cols = np.where(samples < 0)
    samples[rows, cols] = 0

    # Convert to integers
    samples = samples.astype(int)

    return samples


def identify_coating_periods(x, threshold=450, window=3, plot=False):
    """
    Description
    -----------
        This method uses data from the exit end pyrometer to infer whether
            or not coatings are being conducted. It is based on the
            hypothesis that, when the exit end temperature drops below
            a threshold value (450 degrees C by default), coating is
            taking place. The parameter 'window' allows us to assign points
            before and after the temperature drop as belonging to a coating
            period.

    Parameters
    ----------
        x : Data from 7532 Exit End Pyrometer Temperature Centre (PV).

        threshold : drops below this temperature are considered an indication
            that coating is taking place.

        window : the number of points before and after a cooler period that
            we also include in the coating period.

        plot : optional plot of results.

    Returns
    -------
        coating_ind : numpy array of boolean values, where True indicates
            that a point should be considered part of a coating period.

    """

    # Create array of boolean values indicating where coating is taking
    # place. Note that by default we initially consider the first 'window'
    # points as being False.
    N = len(x)
    coating_ind = np.repeat(False, N)
    for i in range(window, N):
        if x[i] < threshold:
            coating_ind[i-window: i+window] = True

    # Optional plot to illustrate results
    if plot:

        indx = np.arange(0, N)
        fig, ax = plt.subplots()
        ax.plot(indx, x, c='k')
        ax.plot(indx[coating_ind], x[coating_ind], 'o', c='r',
                label='Coating Period')
        ax.legend()
        plt.show()

    return coating_ind
