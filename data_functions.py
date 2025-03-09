import pandas as pd

def macro_date(data):
    data['observation_date'] = pd.to_datetime(data['observation_date'], format='%Y-%m-%d')
    data['year'] = data['observation_date'].dt.year
    data['month'] = data['observation_date'].dt.month
    data.drop(columns='observation_date', inplace=True)
    return data

def quarterly_to_monthly(data):
    data['date'] = pd.to_datetime(data[['year', 'month']].assign(day=1))
    data = data.set_index('date').resample('ME').ffill().reset_index()
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data.drop(columns='date', inplace=True)
    return data

def yearly_change(data, name, new_name):
    data[f'{new_name}_yr_change'] = data[f'{name}'].diff(12)
    data.dropna(inplace=True)
    data.drop(columns=f'{name}', inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data