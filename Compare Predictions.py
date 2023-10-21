# Compare predictions between models
correct_in_svm_not_in_dt = np.sum((svm_predictions == y_test) & (dt_predictions != y_test))
correct_in_dt_not_in_svm = np.sum((dt_predictions == y_test) & (svm_predictions != y_test))
incorrect_in_both = np.sum((svm_predictions != y_test) & (dt_predictions != y_test))
