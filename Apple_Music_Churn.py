import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import learning_curve

df = pd.read_csv("AppleMusic_Churn_Converted.csv")

X = df.drop(columns=['userID', 'churned'])
y = df['churned']

y = y.map({'Yes': 1, 'No': 0})

imputer = SimpleImputer(strategy='most_frequent') 
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

log_reg_pred = log_reg.predict(X_test)
log_reg_conf_matrix = confusion_matrix(y_test, log_reg_pred)
log_reg_class_report = classification_report(y_test, log_reg_pred)

dt_pred = dt.predict(X_test)
dt_conf_matrix = confusion_matrix(y_test, dt_pred)
dt_class_report = classification_report(y_test, dt_pred)

rf_pred = rf.predict(X_test)
rf_conf_matrix = confusion_matrix(y_test, rf_pred)
rf_class_report = classification_report(y_test, rf_pred)

print("Logistic Regression Confusion Matrix:")
print(log_reg_conf_matrix)
print("Logistic Regression Classification Report:")
print(log_reg_class_report)

print("Decision Tree Confusion Matrix:")
print(dt_conf_matrix)
print("Decision Tree Classification Report:")
print(dt_class_report)

print("Random Forest Confusion Matrix:")
print(rf_conf_matrix)
print("Random Forest Classification Report:")
print(rf_class_report)

def plot_roc_curve(model, X_test, y_test, label):
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')

plt.figure(figsize=(10, 6))
plot_roc_curve(log_reg, X_test, y_test, 'Logistic Regression')
plot_roc_curve(dt, X_test, y_test, 'Decision Tree')
plot_roc_curve(rf, X_test, y_test, 'Random Forest')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.show()

def plot_precision_recall_curve(model, X_test, y_test, label):
    y_scores = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    average_precision = average_precision_score(y_test, y_scores)
    plt.plot(recall, precision, label=f'{label} (AP = {average_precision:.2f})')

plt.figure(figsize=(10, 6))
plot_precision_recall_curve(log_reg, X_test, y_test, 'Logistic Regression')
plot_precision_recall_curve(dt, X_test, y_test, 'Decision Tree')
plot_precision_recall_curve(rf, X_test, y_test, 'Random Forest')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison')
plt.legend(loc='lower left')
plt.show()

def plot_feature_importance(model, X_train, label):
    feature_importances = model.feature_importances_
    feature_names = X.columns

    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(f'{label} Feature Importance')
    plt.show()

plot_feature_importance(rf, X_train, 'Random Forest')
plot_feature_importance(dt, X_train, 'Decision Tree')

def plot_learning_curve(model, X_train, y_train, label):
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, n_jobs=-1)
    plt.plot(train_sizes, train_scores.mean(axis=1), label=f'{label} - Train')
    plt.plot(train_sizes, test_scores.mean(axis=1), label=f'{label} - CV')

plt.figure(figsize=(10, 6))
plot_learning_curve(log_reg, X_train, y_train, 'LogReg')
plot_learning_curve(dt, X_train, y_train, 'DecisionTree')
plot_learning_curve(rf, X_train, y_train, 'RandomForest')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve Comparison')
plt.legend(loc='best')
plt.show()

def churn_action(prediction_prob):
    if prediction_prob > 0.7:
        return "Send discount email"
    elif prediction_prob > 0.5:
        return "Send personalized music recommendation"
    else:
        return "No action"

for i in range(5):
    user_data = X_test[i:i+1]
    churn_prob = rf.predict_proba(user_data)[:, 1]
    action = churn_action(churn_prob[0])
    print(f"User {i} predicted churn probability: {churn_prob[0]:.2f} - Action: {action}")