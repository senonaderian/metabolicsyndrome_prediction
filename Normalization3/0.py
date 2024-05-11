import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

data = pd.read_csv('ex.csv')
data = data.apply(pd.to_numeric, errors='coerce')

target = data['metabolicsyndrome']
data = data.drop('metabolicsyndrome', axis=1)

std_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
robust_scaler = RobustScaler()

std_data = std_scaler.fit_transform(data)
minmax_data = minmax_scaler.fit_transform(data)
robust_data = robust_scaler.fit_transform(data)

std_df = pd.DataFrame(std_data, columns=data.columns)
std_df['metabolicsyndrome'] = target

minmax_df = pd.DataFrame(minmax_data, columns=data.columns)
minmax_df['metabolicsyndrome'] = target

robust_df = pd.DataFrame(robust_data, columns=data.columns)
robust_df['metabolicsyndrome'] = target

std_df.to_csv('ex2.csv', index=False)
minmax_df.to_csv('ex1.csv', index=False)
robust_df.to_csv('ex3.csv', index=False)
