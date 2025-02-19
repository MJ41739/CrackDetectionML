from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
print(confusion_matrix(y_test.argmax(axis=1), y_pred_labels))
print(classification_report(y_test.argmax(axis=1), y_pred_labels))
