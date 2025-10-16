
import pandas as pd

df = pd.read_csv('tsla_2014_2023.csv')
df.drop(columns=['date', 'next_day_close']).mean().to_json('feature_defaults.json')
