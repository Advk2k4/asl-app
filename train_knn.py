import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

def train_knn():
    df = pd.read_csv("asl_landmarks.csv")
    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    joblib.dump(model, "asl_knn_model.joblib")
    print("✅ Model saved as asl_knn_model.joblib")

if __name__ == "__main__":
    train_knn()
