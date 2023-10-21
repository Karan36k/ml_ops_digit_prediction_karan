from sklearn.metrics import f1_score

svm_f1 = f1_score(y_test, svm_predictions, average='macro')
dt_f1 = f1_score(y_test, dt_predictions, average='macro')
