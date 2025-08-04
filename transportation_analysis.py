import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ----------------------------
# 1️⃣ Load & Clean Dataset
# ----------------------------
def load_and_clean_data(file_path):
    print("Loading data...")
    df = pd.read_excel(file_path)

    # Fill missing values
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna('Unknown', inplace=True)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip().str.title()

    # Remove outliers (IQR method)
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df[col] = np.where((df[col] < lower) | (df[col] > upper), df[col].median(), df[col])

    print("Data loaded and cleaned.")
    print("Columns:", df.columns.tolist())
    return df

# ----------------------------
# 2️⃣ Exploratory Data Analysis
# ----------------------------
def perform_eda(df):
    print("\n📌 Descriptive Statistics:")
    print(df.describe(include="all"))

    # Histograms
    df.hist(figsize=(12, 6))
    plt.suptitle("Histograms of Numeric Variables")
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(10, 5))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap")
    plt.show()

# ----------------------------
# 3️⃣ Feature Engineering
# ----------------------------
def preprocess_data(df):
    print("Preprocessing data (encoding and scaling)...")
    df_encoded = pd.get_dummies(df, drop_first=True)
    scaler = StandardScaler()
    numeric_cols = df_encoded.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col != "year"]
    
    numeric_cols = [col for col in numeric_cols if col != "number_of_trips"]

    df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
    print("Preprocessing done.")
    return df_encoded
 

# ----------------------------
# 4️⃣ Machine Learning Model
# ----------------------------
def train_clustering_model(df_encoded):
    print("Training clustering model...")

    X = df_encoded.copy()
    scores = {}

    for k in range(2, 8):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            scores[k] = score
            print(f"🔸 K={k}, Silhouette Score={score:.3f}")
        except Exception as ex:
            print(f"❌ Failed at K={k}: {ex}")

    if not scores:
        raise ValueError("No clustering configuration succeeded.")

    best_k = max(scores, key=scores.get)
    print(f"✅ Best K found: {best_k}")

    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    X["cluster"] = final_model.fit_predict(X)
    print("Cluster counts:\n", X["cluster"].value_counts())

    return final_model, X

# ----------------------------
# 5️⃣ Recommend Best Routes
# ----------------------------
def recommend_best_routes(df_clustered, top_n=3):
    print("Recommending best routes per cluster...")
    if "cluster" not in df_clustered.columns:
        raise ValueError("Dataset must have a 'cluster' column from KMeans clustering.")

    if "route" not in df_clustered.columns or "number_of_passengers" not in df_clustered.columns:
        raise ValueError("Dataset must have 'route' and 'number_of_passengers' columns.")

    recommendations = {}

    grouped = df_clustered.groupby(["cluster", "route"])["number_of_passengers"].mean().reset_index()

    for cluster_id in sorted(df_clustered["cluster"].unique()):
        top_routes = (
            grouped[grouped["cluster"] == cluster_id]
            .sort_values(by="number_of_passengers", ascending=False)
            .head(top_n)
        )
        recommendations[cluster_id] = top_routes

    print("Recommendation complete.")
    return recommendations

# ----------------------------
# 6️⃣ Main Function
# ----------------------------
def main():
    try:
        print("🔵 Starting main()") 
        # Load & clean data
        df = load_and_clean_data("C:/Users/Julienne Ayinkamiye/Downloads/transportation.xlsx")
        print("✅ Data loaded. Shape:", df.shape)

        # Perform EDA
        print("🔵 Performing EDA...")
        perform_eda(df) 
        # Preprocessing for ML
        print("🔵 Preprocessing data...")
        df_encoded = preprocess_data(df)
        print("✅ Data preprocessed. Shape:", df_encoded.shape) 
        # Train clustering model
        print("🔵 Training clustering model...")
        model, df_clustered = train_clustering_model(df_encoded)
        print("✅ Clustering complete. Columns:", df_clustered.columns) 
        # Add route & passengers back
        print("🔵 Adding back route and passenger columns...")
        print("Route in df? ", "route" in df.columns)
        print("Number of passengers in df? ", "number_of_passengers" in df.columns) 
        if "route" in df.columns and "number_of_passengers" in df.columns:
            df_clustered["route"] = df["route"].values
            df_clustered["number_of_passengers"] = df["number_of_passengers"].values
        else:
            raise KeyError("Missing route or number_of_passengers")
 
        print("🔵 Recommending best routes...")
        recommendations = recommend_best_routes(df_clustered, top_n=3)

        for cluster, routes in recommendations.items():
            print(f"\n🚍 Best Routes for Cluster {cluster}:")
            print(routes)
 
        output_path = "C:/Users/Julienne Ayinkamiye/Downloads/cleaned_clustered_transportation.csv"
        df_clustered.to_csv(output_path, index=False)
        print(f"\n✅ Final dataset saved to: {output_path}")

    except Exception as e:
        print("\n❌ ERROR OCCURRED:")
        print(e)
# ----------------------------
# Run the script
# ----------------------------
if __name__ == "__main__":
    main()
