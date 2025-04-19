from sklearn import metrics 
from sklearn.model_selection import train_test_split  
import pandas as pd
import numpy  as np


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


df = pd.read_csv('anomaly.csv')
print(df.head())

print(df.groupby('Quality')['Quality'].count)

df.drop(['Date'], axis=1, inplace=True)

# If there are missing entries, drop them 
df.dropna(inplace=True, axis=1)

df.Quality[df.Quality == 'Good'] = 1
df.Quality[df.Quality == 'Bad'] = 2

good_mask = df['Quality'] == 1 
bad_mask = df['Quality'] == 2

df.drop('Quality', axiz=1, inplace=True)

df_good = df[good_mask]
df_bad = df[bad_mask]
print(df_bad.head())

print(f"Good Count: {len(de_good)}")
print(f"Bad Count: {len(de_bad)}")

x_good = df_good.values
x_bad = df_bad.values

x_good_train, x_good_test = train_test_split(x_good, test_size = 0.25, random_state=42)

print(f"Good Count: {len(de_good)}")
print(f"Bad Count: {len(de_bad)}")

   

# My Model 
model = Sequential()
model.add(Dense(10, input_dim=x_good.shape[1], activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(x_good.shape[1]))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()


model.fit(x_good_train, x_good_train, verbose=1, epochs=100)


pred = model.predict(x_good_test)
score1 = np.sqrt(metrics.mean_squared_error(pred,x_good_test))

pred = model.predict(x_good_test)
score2 = np.sqrt(metrics.mean_squared_error(pred,x_good_test))

pred = model.predict(x_good_test)
score3 = np.sqrt(metrics.mean_squared_error(pred,x_good_test))
print(f"Insample Good Score (RSME) : {score1}".format(score1))
print(f"Out of Sample Good Score (RSME) : {score2}")
print(f"Bad sample Score (RSME) : {score3}")
