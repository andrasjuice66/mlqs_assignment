import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, balanced_accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from xgboost import plot_importance, plot_tree
import time
import seaborn as sns
def xgboost_train_test(train_df, test_df):
    # Encode the target labels for the training set
    le = LabelEncoder()
    train_df = train_df.copy()
    test_df = test_df.copy()
         
    train_df['Activity'] = le.fit_transform(train_df['Activity'])
    test_df['Activity'] = le.transform(test_df['Activity'])

    # Reorder columns to make 'Activity' the first column
    first_column = 'Activity'
    train_cols = [first_column] + [col for col in train_df.columns if col != first_column]
    test_cols = [first_column] + [col for col in test_df.columns if col != first_column]
    train_df = train_df[train_cols]
    test_df = test_df[test_cols]

    # Convert object columns to categorical or numerical if possible
    for col in train_df.columns:
        if train_df[col].dtype == 'object':
            try:
                train_df[col] = pd.to_numeric(train_df[col])
                test_df[col] = pd.to_numeric(test_df[col])
            except ValueError:
                train_df[col] = train_df[col].astype('category')
                test_df[col] = test_df[col].astype('category')

    # Split the data into features and target
    X_train = train_df.drop(columns=['Activity'])
    y_train = train_df['Activity']
    X_test = test_df.drop(columns=['Activity'])
    y_test = test_df['Activity']

    print(X_train.columns, X_test.columns)

    # Initialize the XGBoost classifier
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, enable_categorical=True)

    # Train the model
    xgb.fit(X_train, y_train)

    # Make predictions
    y_pred = xgb.predict(X_test)
    y_pred_proba = xgb.predict_proba(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("F1-Score:", f1_score(y_test, y_pred, average='weighted'))
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # ROC-AUC Score (One-vs-Rest)
    roc_auc_ovr = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    print("ROC-AUC Score (One-vs-Rest):", roc_auc_ovr)

    # Extract feature importance scores
    importance_scores = xgb.get_booster().get_score(importance_type='weight')
    # Sort by importance
    importance_df = pd.DataFrame(list(importance_scores.items()), columns=['Feature', 'Importance'])
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    # Get the top 10 features
    top_10_features = importance_df.head(10)
    print("Top 10 important features:")
    top_10_features_list = top_10_features['Feature'].tolist()
    print(top_10_features_list)

    
    # Plot feature importance
    plt.figure(figsize=(12, 10))  # Increase the figure size for better readability
    ax = plot_importance(xgb, max_num_features=10)
    plt.title('Top 10 Feature Importances')

    # Rotate the x-axis labels for better visibility
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
    plt.show()

    return top_10_features_list


def main():
    print("Model trainings start...")
    start_time = time.time()
    print('Read training dataset')
    train_df = pd.read_csv('data_agg/feature_engineered_train.csv', index_col=0)
    print('Read testing dataset')
    test_df = pd.read_csv('data_agg/feature_engineered_test.csv', index_col=0)
    end_time_data = time.time()
    print(f"Datasets read ended in {end_time_data - start_time:.2f} seconds.")
    print('Start with XGBoost')
    sel_features = xgboost_train_test(train_df, test_df)
    end_time_xgb = time.time()
    print(f"XGBoost training and testing ended in {end_time_xgb - end_time_data:.2f} seconds.")

    # Train on selected features
    print("Training start on selected features")
    sel_features_with_activity = sel_features + ['Activity']
    sel_features2 = xgboost_train_test(train_df[sel_features_with_activity], test_df[sel_features_with_activity])
    print(sel_features2)

if __name__ == "__main__":
    main()
