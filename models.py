from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score

# Cargar el conjunto de datos de iris
iris = load_iris()
X, y = iris.data, iris.target

# Dividir los datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Crear el modelo de SVM
clf = SVC()

# Entrenar el modelo
clf.fit(X_train, y_train)

# Hacer predicciones en los datos de prueba
y_pred = clf.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Guardar el modelo entrenado
import joblib
joblib.dump(clf, 'svm_iris_model.joblib')



from sklearn.datasets import make_moons

# Generar el conjunto de datos
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# Crear el modelo de DBSCAN
clustering = DBSCAN(eps=0.3, min_samples=5)

# Entrenar el modelo
clustering.fit(X)

# Hacer predicciones en los datos
y_pred = clustering.fit_predict(X)

# Guardar el modelo entrenado
joblib.dump(clustering, 'dbscan_moons_model.joblib')

