import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer, TargetEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from joblib import dump

# Load data and split
df = pd.read_csv('data/real_estate/synthetic_data.csv')
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Save train and test data to csv
X_train.to_csv('data/train_test/X_train.csv', index=False)
X_test.to_csv('data/train_test/X_test.csv', index=False)
y_train.to_csv('data/train_test/y_train.csv', index=False)
y_test.to_csv('data/train_test/y_test.csv', index=False)

# Define preprocessing
categorical_cols = ['city', 'state', 'brokered_by', 'zip_code']
numerical_cols = ['bed', 'bath', 'acre_lot', 'house_size']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', TargetEncoder(), categorical_cols),
        ('num', QuantileTransformer(output_distribution='normal'), numerical_cols)
    ],
    remainder='passthrough'
)

# Create individual pipelines
models = {
    'random_forest': RandomForestRegressor(verbose=2, n_estimators=100, n_jobs=-1),
    'xgboost': XGBRegressor(verbosity=2, n_estimators=100),
}

# Train and save models
for model_name, model in models.items():
    # Create full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Train
    pipeline.fit(X_train, y_train)
    
    # Save entire pipeline (including preprocessing)
    dump(pipeline, f'models/{model_name}_pipeline.joblib')
    
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mape = 100 * abs((y_test - y_pred) / y_test).mean()

    print(f'{model_name} R2: {r2:.4f}, MAPE: {mape:.2f}%')
    print(f'{model_name} saved to models/{model_name}_pipeline.joblib')