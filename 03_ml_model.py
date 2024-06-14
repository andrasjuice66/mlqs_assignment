import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from xgboost import plot_importance
import time

def xgboost_train_test(train_df, test_df):
    # Encode the target labels for the training set
    le = LabelEncoder()
    train_df['Activity'] = le.fit_transform(train_df['Activity'])
    test_df['Activity'] = le.transform(test_df['Activity'])

    # Reorder columns to make 'Activity' the first column
    first_column = 'Activity'
    train_cols = [first_column] + [col for col in train_df.columns if col != first_column]
    test_cols = [first_column] + [col for col in test_df.columns if col != first_column]
    train_df = train_df[train_cols]
    test_df = test_df[test_cols]

    # Split the data into features and target
    X_train = train_df.drop(columns=['Activity'])
    y_train = train_df['Activity']
    X_test = test_df.drop(columns=['Activity'])
    y_test = test_df['Activity']

    # Initialize the XGBoost classifier
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

    # Train the model
    xgb.fit(X_train, y_train)

    # Make predictions
    y_pred = xgb.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Plot feature importance
    plot_importance(xgb)
    plt.show()

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
    xgboost_train_test(train_df, test_df)
    end_time_xgb = time.time()
    print(f"XGBoost training and testing ended in {end_time_xgb - end_time_data:.2f} seconds.")

if __name__ == "__main__":
    main()
