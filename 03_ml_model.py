import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from xgboost import plot_importance
import time

def xgboost_train_test(df):
    df = df.iloc[:int(0.2 * len(df))]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    le = LabelEncoder()
    df['Activity'] = le.fit_transform(df['Activity'])
    first_column = 'Activity'
    cols = [first_column] + [col for col in df.columns if col != first_column]
    df = df[cols]
    X = df.drop(columns=['Activity'])
    y = df['Activity']
    
    # Define the split index
    split_index = int(0.8 * len(df))
    
    # Split the data into training and testing sets
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    
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
    print('Read dataset')
    df = pd.read_csv('data_agg/feature_engineered.csv', index_col=0)
    end_time_data = time.time()
    print(f"Dataset read ended in {end_time_data - start_time:.2f} seconds.")
    print('Start with XGBoost')
    xgboost_train_test(df)
    end_time_xgb = time.time()
    print(f"XGBoost training and testing ended in {end_time_xgb - end_time_data:.2f} seconds.")

if __name__ == "__main__":
    main()
