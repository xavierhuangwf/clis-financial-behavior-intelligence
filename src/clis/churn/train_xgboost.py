import labelPreparation as lp
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns


def get_XGBoost(X, y):
    pos_weight = sum(y == 0) / sum(y == 1)
    xTrain, xVal, yTrain, yVal = train_test_split(X, y, test_size=0.2, random_state=42)
    xgb_model = XGBClassifier(objective='binary:logistic', eval_metric="logloss", scale_pos_weight=pos_weight)

    param_dist = {
        'max_depth': np.arange(3, 8, 1),
        'n_estimators': np.arange(50, 300, 50),
        'learning_rate': np.linspace(0.01, 0.1, 5),
        'subsample': np.linspace(0.5, 0.9, 5),
        'colsample_bytree': np.linspace(0.5, 0.9, 5)
    }
    # use randomized search to find the best hyperparameters
    random_search = RandomizedSearchCV(xgb_model, param_distributions=param_dist,
                                       n_iter=50, scoring='f1', cv=5, verbose=1, n_jobs=-1)
    random_search.fit(xTrain, yTrain)

    print("Best Parameters:", random_search.best_params_)
    print("Best F1 score:", random_search.best_score_)

    best_xgb_model = random_search.best_estimator_
    print("Importance: ", best_xgb_model.feature_importances_)

    # the eval_set parameter is to monitor the performance(logloss) of the model on training and validation data and
    # check if this model is overfitting (how does logloss change, is this metric on training and validation data
    # close to each other?)
    eval_set = [(xTrain, yTrain), (xVal, yVal)]
    best_xgb_model.fit(xTrain, yTrain, eval_set=eval_set, verbose=False)

    y_train_pred = best_xgb_model.predict(xTrain)
    y_val_pred = best_xgb_model.predict(xVal)

    train_f1 = f1_score(yTrain, y_train_pred)
    val_f1 = f1_score(yVal, y_val_pred)

    print(f"Train F1 Score: {train_f1:.4f}")
    print(f"Validation F1 Score: {val_f1:.4f}")

    results = best_xgb_model.evals_result()
    epochs = len(results['validation_0']['logloss'])

    plt.figure(figsize=(8, 5))
    plt.plot(range(epochs), results['validation_0']['logloss'], label='Train Loss')
    plt.plot(range(epochs), results['validation_1']['logloss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Log Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    plt.savefig('./figures/Training vs Validation Loss.png')
    plt.show()

    if train_f1 - val_f1 > 0.1:
        print("Overfitting Warning!")
    return best_xgb_model


file_path = "../../simulated_fake_transactions_dataset_2.csv"
features_labels = lp.get_features_labels(file_path)
features = features_labels.iloc[:, 3:9]
labels = features_labels['churn_label']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
xgb = get_XGBoost(X_train, y_train)
y_pred = xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy: ", accuracy)
print("F1: ", f1)

y_proba = xgb.predict_proba(X_test)[:, 1]
plt.figure(figsize=(8, 5))
sns.histplot(y_proba, bins=30, kde=True, color='blue', alpha=0.6)

plt.title("Churn Probability Distribution", fontsize=14)
plt.xlabel("Predicted Probability", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.grid(True)
plt.savefig('./figures/Churn Probability Distribution.png')
plt.show()

xgb.save_model('xgboost_model.json')
