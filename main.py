 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as knn


df = pd.read_csv(r"C:\Users\haddi\OneDrive\Desktop\cpp\python\chrun.csv")
print(df.head(10))
X=df[['MonthlyCharges','tenure','PaymentMethod','Contract']]
y=df['Churn']

#Swap the text to Numbers (0,1)
X=pd.get_dummies(X,drop_first=True)
print(X.head())

#Split data (80% training , 20% testing)
X_train , X_test , y_train , y_test =train_test_split(X ,y , test_size=0.2 , random_state=42 )

#Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Model Training
clf = knn(n_neighbors=30)
clf.fit(X_train , y_train)

#Test
accuracy = clf.score(X_test , y_test)
print(f"Accuracy: {accuracy*100:.2f}%")

#----------------------------------------
#loop for trying new values of k 
import matplotlib.pyplot as plt
accuracies = []
k_values = range(1,41)

for k in k_values:
    clf= knn(n_neighbors=k)
    clf.fit(X_train , y_train)
    acc = clf.score(X_test, y_test)
    accuracies.append(acc)

#Visualisation

plt.figure(figsize=(10, 5))
plt.plot(k_values , accuracies , marker='o' , color = 'red' , linestyle='dashdot') #k_values (X axis) accuracies (y axis)
plt.title("KNN: Accuracy Vs K Value")
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.grid()
plt.show()  



    
       

