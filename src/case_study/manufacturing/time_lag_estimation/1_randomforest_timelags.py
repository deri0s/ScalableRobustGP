import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Choose the scanner where the fault density data is read from
scanner = 'ISRA'

# Choose the furnace inputs
to_retain = ['10091 Furnace Load',
             '10271 C9 (T012) Upstream Refiner',
             '2922 Closed Bottom Temperature - Downstream Working End (PV)',
             '2921 Closed Bottom Temperature - Upstream Working End (PV)',
             '2918 Closed Bottom Temperature - Port 6 (PV)',
             '2923 Filling Pocket Closed Bottom Temperature Centre (PV)',
             '7546 Open Crown Temperature - Port 1 (PV)',
             '7746 Open Crown Temperature - Port 2 (PV)',
             '7522 Open Crown Temperature - Port 4 (PV)',
             '7483 Open Crown Temperature - Port 6 (PV)',
             '7520 Open Crown Temperature - Upstream Working End (PV)',
             '9400 Port 2 Gas Flow (SP)',
             '9282 Tweel Position',
             '11384 Wobbe Index (Incoming Gas)']

# Choose the standardisation method
# minmax = normalise, standard = standardise
stand_method = 'normalise'

# ----------------------------------------------------------------------------
# LOAD DATA FOR TRAINING AND TESTING
# ----------------------------------------------------------------------------

# Initialise empty data frames
X_df, Y_df, Y_raw_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Loop over available files of post-processed data
for i in range(1, 5):
    file_name = 'Input Post-Processing ' + str(i) + ' ' + scanner + '.xlsx'
    X_df = X_df._append(pd.read_excel(file_name, sheet_name='input_data'))
    Y_df = Y_df._append(pd.read_excel(file_name, sheet_name='output_data'))
    Y_raw_df = Y_raw_df._append(pd.read_excel(file_name, sheet_name='raw_output_data'))

# ----------------------------------------------------------------------------
# REMOVE INPUTS WE ARE NOT GOING TO USE
# ----------------------------------------------------------------------------

input_names = X_df.columns
for name in input_names:
    if name not in to_retain:
        X_df.drop(columns=name, inplace=True)

# Check that the data frames contain the correct number of inputs
assert len(X_df.columns) == len(to_retain)

# Check that the data frame input names match those in to_retain
assert set(X_df.columns) == set(to_retain)

# Standardise input data
if stand_method == 'standardise':
    scaler = StandardScaler()
    X_df_stand = scaler.fit_transform(X_df)
    X_df_standardised = pd.DataFrame(X_df_stand, columns=X_df.columns)
elif stand_method == 'normalise':
    scaler2 = MinMaxScaler(feature_range=(0, 1))
    X_df_norm = scaler2.fit_transform(X_df)
    X_df_normalised = pd.DataFrame(X_df_norm, columns=X_df.columns)
else:
    raise ValueError("Invalid standardisation method. Choose 'normalise' or 'standardise'.")

# ----------------------------------------------------------------------------
# CREATE LAGGED FEATURES
# ----------------------------------------------------------------------------

def create_lagged_features(df, features, lags):
    for feature in features:
        for lag in range(1, lags + 1):
            df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
    return df

# Example usage
lags = 288
X_df = create_lagged_features(X_df_normalised, to_retain, lags)
X_df.dropna(inplace=True)
Y_df = Y_df.iloc[lags:].reset_index(drop=True)
X_df = X_df.reset_index(drop=True)

# ----------------------------------------------------------------------------
# SPLIT DATA INTO TRAINING AND TESTING SETS
# ----------------------------------------------------------------------------

y_cleaned = pd.read_csv('validation_data.csv')

start_train = Y_df[Y_df['Time stamp'] == y_cleaned.loc[0, 'date_time']].index[0]
end_train = Y_df[Y_df['Time stamp'] == '2020-08-29'].index[0]

X_train = X_df[start_train:end_train]
N_train = end_train - start_train
y_train = y_cleaned.loc[:N_train-1, 'gp_pred']

y_test = y_cleaned.loc[:, 'gp_pred']
dif = len(y_test) - N_train

end_test = end_train + dif
X_test = X_df[start_train:end_test]
date_time = Y_df.loc[start_train:end_test-1, 'Time stamp']
y_raw = Y_raw_df[start_train:end_test]

# ----------------------------------------------------------------------------
# TRAIN THE RANDOM FOREST MODEL
# ----------------------------------------------------------------------------

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ----------------------------------------------------------------------------
# EVALUATE FEATURE IMPORTANCE
# ----------------------------------------------------------------------------

importances = model.feature_importances_
feature_names = X_train.columns

feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

# ----------------------------------------------------------------------------
# ANALYZE THE RESULTS
# ----------------------------------------------------------------------------

print(feature_importance_df.head(10))

feature_importance_df.head(40)

fi = feature_importance_df.copy()
primera = fi.iloc[0,0]

codes = []
lags = []
codes.append(primera.split()[0])
lags.append(primera.split()[-1])

for i in fi.iloc[0:200,0]:
    code = i.split()[0]
    lag = i.split()[-1]
    if code in codes:
        pass
    else:
        codes.append(code)
        lags.append(lag.split("_")[-1])

lags[0] = 29
d = {code:lag for code,lag in zip(codes,lags)}
print('Estimated time lags: \n', d)

# ----------------------------------------------------------------------------
# VISUALIZE FEATURE IMPORTANCE
# ----------------------------------------------------------------------------

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df.loc[0:10, 'feature'],
         feature_importance_df.loc[0:10, 'importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()

fig, ax = plt.subplots()
# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
fig.autofmt_xdate()

plt.axvline(date_time.iloc[N_train-1], linestyle='--', linewidth=3,
            color='black')
ax.plot(date_time, y_test, label='filtered')
ax.plot(date_time, y_pred, label='Random-Forest')
plt.show()