# ---------------------------------------------------------
# MACHINE LEARNING PROJECT – CLASSIFICATION (KNN, SVM, DT, RF)
# DATASET: Breast Cancer Wisconsin
# ---------------------------------------------------------

# IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# ------------------------------------------
# LOAD DATASET
# ------------------------------------------
data=load_breast_cancer()
X=pd.DataFrame(data.data,columns=data.feature_names)
y=pd.Series(data.target)

print("Shape of X:",X.shape)
print("Shape of y:",y.shape)

# ------------------------------------------
# TRAIN-TEST SPLIT
# ------------------------------------------
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

# ------------------------------------------
# FEATURE SCALING
# ------------------------------------------
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# ------------------------------------------
# MODEL 1: KNN
# ------------------------------------------
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
knn_pred=knn.predict(X_test)
acc_knn=accuracy_score(y_test,knn_pred)

# ------------------------------------------
# MODEL 2: SVM
# ------------------------------------------
svm=SVC(kernel='rbf')
svm.fit(X_train,y_train)
svm_pred=svm.predict(X_test)
acc_svm=accuracy_score(y_test,svm_pred)

# ------------------------------------------
# MODEL 3: DECISION TREE
# ------------------------------------------
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
dt_pred=dt.predict(X_test)
acc_dt=accuracy_score(y_test,dt_pred)

# ------------------------------------------
# MODEL 4: RANDOM FOREST
# ------------------------------------------
rf=RandomForestClassifier(n_estimators=100)
rf.fit(X_train,y_train)
rf_pred=rf.predict(X_test)
acc_rf=accuracy_score(y_test,rf_pred)

# ------------------------------------------
# ACCURACY COMPARISON
# ------------------------------------------
results=pd.DataFrame({
    "Model":["KNN","SVM","Decision Tree","Random Forest"],
    "Accuracy":[acc_knn,acc_svm,acc_dt,acc_rf]
})

print("\n---MODEL ACCURACY RESULTS---")
print(results)

# ------------------------------------------
# CONFUSION MATRIX (BEST MODEL)
# ------------------------------------------
best_model_name=results.sort_values("Accuracy",ascending=False).iloc[0]["Model"]
print("\nBest Model:",best_model_name)

best_model={
    "KNN":knn,
    "SVM":svm,
    "Decision Tree":dt,
    "Random Forest":rf
}[best_model_name]

cm=confusion_matrix(y_test,best_model.predict(X_test))

plt.figure(figsize=(6,4))
plt.imshow(cm,cmap='Blues')
plt.title(f"Confusion Matrix-{best_model_name}")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ------------------------------------------
# CLASSIFICATION REPORT
# ------------------------------------------
print("\n---Classification Report---")
print(classification_report(y_test,best_model.predict(X_test)))
