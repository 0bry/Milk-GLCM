import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

df = pd.read_csv('ekstraksi_fitur\glcm_features_ex.csv')

X = df.drop('class', axis=1)
y = df['class']

k_best = 20  
selector = SelectKBest(score_func=f_classif, k=k_best)
X_new = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]

new_df = df[['class'] + list(selected_features)]

new_df.to_csv('selected_glcm_features.csv', index=False)