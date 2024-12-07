import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_validate, StratifiedKFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import KFold

# Caricamento dati
df = pd.read_csv('Dataset/train.csv', sep=',')
df_synthetic = pd.read_csv('synthetic_data.csv', sep=',')
df_test = pd.read_csv('Dataset/test.csv', sep=',')

# Concatenazione dei dati
df = pd.concat([df, df_synthetic], axis=0, ignore_index=True)

# Preprocessing del test set
df_test = df_test.rename(columns={'Unnamed: 0': 'row number'})
df_test['diagnosis'] = df_test['diagnosis'].replace('unknown', np.nan).replace('Acute megakaryoblastic leukaemia', np.nan)
df_test['diagnosis'] = df_test['diagnosis'].fillna(value=df_test['diagnosis'].mode().iloc[0])

# Rimozione features costanti
for col in df.columns:
    if len(df[col].unique()) == 1:
        df = df.drop(columns=[col])
        if col in df_test.columns:
            df_test = df_test.drop(columns=[col])

# Imputazione valori mancanti
df['response_chemo'] = df['response_chemo'].fillna('Unknown')
df_test['response_chemo'] = df_test['response_chemo'].fillna('Unknown')

# Imputazione con la mediana per variabili numeriche
for col in df.select_dtypes(include='number').columns:
    median = df[col].quantile(0.5)
    df[col] = df[col].fillna(value=median)
    if col in df_test.columns:
        df_test[col] = df_test[col].fillna(value=median)

# Feature engineering
df['ast alt'] = df['ast'] / df['alt']
df_test['ast alt'] = df_test['ast'] / df_test['alt']
df = df.drop(columns=['ast', 'alt'])
df_test = df_test.drop(columns=['ast', 'alt'])

# Preparazione y train
y = np.array([(bool(status), float(surv)) for status, surv in 
              zip(df['vital_status'].map({'Dead': False, 'Alive': True}),
                  pd.to_numeric(df['overall_survival'], errors='coerce'))],
             dtype=[('event', bool), ('time', float)])

# Preparazione X train e test
X = df.drop(columns=['vital_status', 'overall_survival']).copy()
id_test = df_test['id'].values
X_test = df_test.drop(columns=['id', 'row number']).copy()

# Identificazione delle colonne numeriche e categoriche
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns.union(
    X.select_dtypes(include=['bool', 'int64']).columns.difference(numeric_features)
)

# Converti tutte le colonne categoriche in stringhe
for col in categorical_features:
    if col in X.columns:
        X[col] = X[col].astype(str)
    if col in X_test.columns:
        X_test[col] = X_test[col].astype(str)

# Assicuriamo che le colonne siano allineate
common_numeric = [col for col in numeric_features if col in X_test.columns]
common_categorical = [col for col in categorical_features if col in X_test.columns]

# Selezioniamo solo le colonne comuni
X = X[common_numeric + common_categorical]
X_test = X_test[common_numeric + common_categorical]

# Preprocessing
column_transformer = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), common_numeric),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, 
                            handle_unknown='ignore'), common_categorical)
    ])

# Transform dei dati
X_transformed = column_transformer.fit_transform(X)
X_test_transformed = column_transformer.transform(X_test)

# Setup del modello
RANDOM_STATE = 2493
estimator = RandomSurvivalForest(
    n_estimators=100,
    max_depth=None,
    min_samples_split=6,
    min_samples_leaf=3,
    max_features='sqrt',
    n_jobs=-1,
    random_state=RANDOM_STATE
)

# Definizione dello scorer
def c_index_scorer(estimator, X, y):
    risk_scores = estimator.predict(X)
    return concordance_index_censored(y['event'], y['time'], risk_scores)[0]

# Cross-validation
cv = KFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
scores = cross_validate(
    estimator, X_transformed, y, cv=cv,
    scoring={'cindex': c_index_scorer}
)

print(f"cv-folds-scores: {scores['test_cindex']}")
print(f"   cv-est-score: {scores['test_cindex'].mean():.4f}")

# Fit finale e predizione
estimator = estimator.fit(X_transformed, y)
predictions = estimator.predict(X_test_transformed)

# Creazione submission file
submission = pd.DataFrame({
    'id': id_test,
    'outcome': predictions
})
submission = submission.sort_values('id')
submission.to_csv('submission.csv', index=False)

print("\nSubmission file creato con successo!")
print("\nPrime righe del submission file:")
print(submission.head())