import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from xgboost import plot_importance
import time


def xgboost_train_test(df):

    df = df.iloc[:int(0.2 * len(df))]
    print(df.columns)

    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Drop the timestamp column as it's not needed for XGBoost directly
    df = df.drop(columns=['Time (s)'])


    # Encode the target labels
    le = LabelEncoder()
    df['Activity'] = le.fit_transform(df['Activity'])

    # Reorder columns to make 'Activity' the first column
    first_column = 'Activity'
    cols = [first_column] + [col for col in df.columns if col != first_column]
    df = df[cols]
    print(df.head)


    # Split the data into features and target
    X = df.drop(columns=['Activity'])
    y = df['Activity']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the XGBoost classifier
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

    # Train the model
    xgb.fit(X_train, y_train)

    # Make predictions
    y_pred = xgb.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=le.classes_))

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


 

    


    
