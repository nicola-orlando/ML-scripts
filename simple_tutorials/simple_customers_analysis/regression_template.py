df = pd.read_csv('customers_trimmed_set_hard_coded.csv')
df_test = pd.read_csv('customers_trimmed_set_hard_coded.csv')

df = df.dropna()
df_test = df_test.dropna()

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

x_train, x_test, y_train, y_test = train_test_split(df, df.Churn, test_size=0.5, random_state=0)
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
# Evaluate prediction on full dataset and cross check I am able to get back the right labeling 
predictions = model.predict(df)

score = model.score(x_test, y_test)
print(score)
df['predictions'] = predictions
