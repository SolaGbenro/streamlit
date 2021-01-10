import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

penguins = pd.read_csv('penguins_cleaned.csv')

# Ordinal feature encoding
df = penguins.copy()
target = 'species'
encode = ['sex', 'island']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

target_mapper = {name: indx for indx, name in enumerate(penguins.species.unique())}


def target_encode(val):
    return target_mapper[val]


df['species'] = df['species'].apply(target_encode)

# (in)dependent variables
X = df.drop('species', axis=1)
y = df['species']

clf = RandomForestClassifier()
clf.fit(X, y)

# save model
pickle.dump(clf, open('penguins_rf_clf.pkl', 'wb'))

