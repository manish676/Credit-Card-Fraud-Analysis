import  pandas as pd
import  numpy as np
import  matplotlib.pylab as plt
import  seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("credit.csv")
# print(df.info) # to see to all info
# print(df.head())
# print(df.isnull())
# print(df.describe())
# print(df.isnull().sum())

#UNIVARIATE ANALYSIS: Looking at the type of data present in different columns

# cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE', 'default.payment.next.month']
# colors = ['blue', 'green', 'red', 'purple']  # Define different colors for each count plot
# fig, ax = plt.subplots(1, 4, figsize=(25, 5))
#
# for cols, subplots, color in zip(cat_cols, ax.flatten(), colors):
#     sns.countplot(x=df[cols], ax=subplots, color=color)  # Specify color for each count plot
#
# plt.show()


# Vizualizing the imbalance

# yes=(((df['default.payment.next.month']==1).sum())/len(df['default.payment.next.month']))*100
# no=(((df['default.payment.next.month']==0).sum())/len(df['default.payment.next.month']))*100
#
# x=[yes,no]
#
# plt.pie(x,labels=['Yes','No'],colors=['red', 'g'],radius=2,autopct='%1.0f%%')
# plt.title('default.payment.next.month')
# plt.show()

print(df['default.payment.next.month'].value_counts(normalize=True))


X=df.drop('default.payment.next.month',axis=1)
y=df['default.payment.next.month']
print(X,y)

# Assuming X and y are already defined and contain your feature matrix and target variable respectively

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Initializing and training the RandomForestClassifier model
rfc = RandomForestClassifier(random_state=42)
model = rfc.fit(X_train, y_train)

# Making predictions on the test set
predictions = model.predict(X_test)

# Evaluating the model
print("Classification Report:")
print(classification_report(y_test, predictions))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nAccuracy Score:")
print(accuracy_score(y_test, predictions))
