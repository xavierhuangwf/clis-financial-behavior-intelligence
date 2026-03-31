import labelPreparation as lp
import pandas as pd
import xgboost as xgb

model = xgb.XGBClassifier()
model.load_model('xgboost_model.json')

# output: account no, start month, end month, churn label
file_path = "../../simulated_fake_transactions_dataset_2.csv"
data = lp.get_features_labels(file_path)
features = data.iloc[:, 3:9]
churn_labels = model.predict(features)
res = pd.DataFrame(columns=['Account No', 'start_month', 'end_month', 'churn_label'])
res['Account No'] = data['Account No']
res['start_month'] = data['start_month']
res['end_month'] = data['end_month']
res['churn_label'] = churn_labels
print(res)
res.to_csv('churn_prediction.csv', index=False)
