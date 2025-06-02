import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("datos_modelo.csv")
X = df[["velocidad_bala","distancia_bala1","bala2_y","distancia_bala2","mov_derecha"]]
y = df["accion"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)
print(classification_report(y_test, clf.predict(X_test)))
