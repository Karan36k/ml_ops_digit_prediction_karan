from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load and preprocess the data (MNIST)
# Split into training and testing sets

# Hyperparameter tuning for SVM
svm_params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
svm_model = GridSearchCV(SVC(), svm_params, cv=3)
svm_model.fit(X_train, y_train)

# Hyperparameter tuning for Decision Tree
dt_params = {'max_depth': [None, 10, 20, 30]}
dt_model = GridSearchCV(DecisionTreeClassifier(), dt_params, cv=3)
dt_model.fit(X_train, y_train)

# Train the production model (SVM) with the best parameters
svm_model_best = svm_model.best_estimator_
svm_model_best.fit(X_train, y_train)

# Train the candidate model (Decision Tree) with the best parameters
dt_model_best = dt_model.best_estimator_
dt_model_best.fit(X_train, y_train)

# Evaluate accuracy for both models
svm_accuracy = accuracy_score(y_test, svm_model_best.predict(X_test))
dt_accuracy = accuracy_score(y_test, dt_model_best.predict(X_test))

# Generate predictions for both models
svm_predictions = svm_model_best.predict(X_test)
dt_predictions = dt_model_best.predict(X_test)

# Compute a 10x10 confusion matrix for both models
svm_confusion_matrix = confusion_matrix(y_test, svm_predictions)
dt_confusion_matrix = confusion_matrix(y_test, dt_predictions)
