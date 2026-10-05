import pandas as pd
import numpy as np
import pickle
import gzip
import os

# ==========================================
# 1. CONSTANTS & MAPPINGS
# ==========================================
KEY_PARTIES = ['MK_Party', 'ANC', 'DA', 'EFF', 'IFP', 'ActionSA']

PARTY_MAP = {
    'UMKHONTO WESIZWE': 'MK_Party',
    'MK PARTY': 'MK_Party',
    'MKP': 'MK_Party',
    'AFRICAN NATIONAL CONGRESS': 'ANC',
    'DEMOCRATIC ALLIANCE': 'DA',
    'ECONOMIC FREEDOM FIGHTERS': 'EFF',
    'INKATHA FREEDOM PARTY': 'IFP',
    'ACTIONSA': 'ActionSA'
}

# ==========================================
# 2. DATA PREPROCESSING & CLEANING
# ==========================================
def preprocess_and_clean(df, year):
    """
    Standardizes schema, harmonizes party names, groups minor parties into 'Others',
    and computes percentage vote share per ward.
    """
    data = df.copy()
    data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')

    column_renames = {
        'ward_no': 'ward_id',
        'wardno': 'ward_id',
        'party_name': 'party',
        'valid_votes': 'votes',
        'total_valid_votes': 'total_votes'
    }
    data = data.rename(columns=column_renames)

    if 'ward_id' in data.columns:
        data['ward_id'] = data['ward_id'].astype(str).str.extract(r'(\d+)')[0]
        data['ward_id'] = 'Ward_' + data['ward_id'].str.zfill(3)

    if 'party' in data.columns:
        data['party'] = data['party'].astype(str).str.strip().str.upper()
        data['party'] = data['party'].map(lambda p: PARTY_MAP.get(p, 'Others'))

    if 'votes' in data.columns:
        data['votes'] = pd.to_numeric(data['votes'], errors='coerce').fillna(0)

    if {'ward_id', 'party', 'votes'}.issubset(data.columns):
        data_grouped = data.groupby(['ward_id', 'party'], as_index=False)['votes'].sum()
        data_grouped['total_ward_votes'] = data_grouped.groupby('ward_id')['votes'].transform('sum')
        data_grouped['vote_share'] = np.where(
            data_grouped['total_ward_votes'] > 0,
            data_grouped['votes'] / data_grouped['total_ward_votes'],
            0
        )
        data_grouped['year'] = year
        return data_grouped

    return data

# Load raw election datasets
df_2016 = pd.read_csv('2016 Dataset.csv')
df_2021 = pd.read_csv('2021 Dataset.csv')
df_2024 = pd.read_csv('Provincial.csv')

# Clean datasets
clean_2016 = preprocess_and_clean(df_2016, 2016)
clean_2021 = preprocess_and_clean(df_2021, 2021)
clean_2024 = preprocess_and_clean(df_2024, 2024)

# Pivot datasets into ward-level features
def pivot_election_data(df, year_prefix):
    pivoted = df.pivot(index='ward_id', columns='party', values='vote_share').fillna(0)
    pivoted.columns = [f"{col}_{year_prefix}" for col in pivoted.columns]
    return pivoted

pivoted_2016 = pivot_election_data(clean_2016, '2016')
pivoted_2021 = pivot_election_data(clean_2021, '2021')
pivoted_2024 = pivot_election_data(clean_2024, '2024')

# Merge historical data
master_df = pivoted_2016.join(pivoted_2021, how='outer').join(pivoted_2024, how='outer').fillna(0)

# ==========================================
# 3. FEATURE ENGINEERING & SCALING
# ==========================================
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# Build target features (e.g., predicting 2024 MK Party performance based on past shifts)
features = [col for col in master_df.columns if '_2016' in col or '_2021' in col]
X = master_df[features]
y = master_df['MK_Party_2024'] if 'MK_Party_2024' in master_df.columns else master_df.iloc[:, 0]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 4. MODEL TRAINING & EXPORT
# ==========================================
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_scaled, y)

# Save artifacts
with gzip.open('gb_model.pkl.gz', 'wb') as f:
    pickle.dump(gb_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model training complete. Exported 'gb_model.pkl.gz' and 'scaler.pkl'.")
