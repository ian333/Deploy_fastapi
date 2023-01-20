from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Carga el conjunto de datos MNIST
mnist = fetch_openml('mnist_784', parser='auto')
X, y = mnist["data"], mnist["target"]

# Dividir los datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Crear el modelo de árbol de decisiones
clf = DecisionTreeClassifier(random_state=0)

# Entrenar el modelo
clf.fit(X_train, y_train)

# Hacer predicciones en los datos de prueba
y_pred = clf.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Guardar el modelo entrenado
import joblib
joblib.dump(clf, 'mnist_model.pkl')
