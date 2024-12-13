from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import pandas as pd

# Step 1: Load datasets
food_df = pd.read_csv('food.csv')  
ratings_df = pd.read_csv('ratings.csv')  # Dataset rating (ratings.csv)

print("\nCek Data Ratings")
print(ratings_df.isnull().sum())
ratings = ratings_df.dropna()
print("\nBanyak Data Ratings Sebelum di Hapus", ratings_df.shape)
print("Banyak Data Ratings Setelah di Hapus", ratings.shape)
print("\nCek Data Ratings Setelah di Hapus")
print(ratings.isnull().sum())

print("\nCek Data Makanan")
print(food_df.isnull().sum())
food = food_df.dropna()
print("\nBanyak Data Makanan", food.shape)

# Step 2: Merge the datasets
merged_df = pd.merge(ratings_df, food_df, on='Food_ID', how='inner')
print("\nKolom setelah di Merge:")
print(merged_df.columns)

print("\nBanyak Kolom setelah di Merge:")
print(merged_df.shape)

# Step 3: Pivot data to create a User-Food matrix
user_food_matrix = merged_df.pivot_table(index='User_ID', columns='Food_ID', values='Rating', fill_value=0)

# Step 4: Calculate cosine similarity between users
user_similarity = cosine_similarity(user_food_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_food_matrix.index, columns=user_food_matrix.index)

print("\nUser Similarity Matrix:")
print(user_similarity_df)

# Step 5: Recommendation function for new users
def recommend_for_new_user(new_user_preferences, user_food_matrix, user_similarity_df, food_df, top_n=5):
    """
    Recommends top N food items for a new user based on preferences and similar users' ratings.
    """
    filtered_food = food_df[
        (food_df['C_Type'].str.strip().str.capitalize().isin(new_user_preferences['categories'])) &
        (food_df['Veg_Non'].str.strip().str.lower() == new_user_preferences['veg_non'])
    ]

    if filtered_food.empty:
        return "Tidak ada makanan yang sesuai dengan preferensi Anda.", pd.DataFrame()
    
    filtered_food_ids = filtered_food['Food_ID'].values
    valid_food_ids = [fid for fid in filtered_food_ids if fid in user_food_matrix.columns]
    if not valid_food_ids:
        return "Tidak ada makanan untuk direkomendasikan berdasarkan preferensi Anda.", pd.DataFrame()
    
    average_user_preferences = user_food_matrix.mean(axis=1)
    most_similar_user = average_user_preferences.idxmax()
    
    similar_user_ratings = user_food_matrix.loc[most_similar_user]
    similar_user_ratings = similar_user_ratings[valid_food_ids]
    
    recommendations = similar_user_ratings.sort_values(ascending=False).head(top_n)
    
    if recommendations.empty:
        return "Tidak ada makanan untuk direkomendasikan berdasarkan pengguna serupa.", pd.DataFrame()
    
    recommended_food = food_df[food_df['Food_ID'].isin(recommendations.index)]
    return recommended_food[['Food_ID', 'Name']], recommendations

# Step 6: Evaluation function 
def evaluate_recommendation_user_based(recommendations, user_preferences, food_df):
    """
    Evaluates recommendation performance per user based on their input preferences.
    """
    recommended_items = recommendations['Food_ID'].values
    filtered_food = food_df[
        (food_df['C_Type'].str.strip().str.capitalize().isin(user_preferences['categories'])) &
        (food_df['Veg_Non'].str.strip().str.lower() == user_preferences['veg_non'])
    ]
    
    true_positive_items = filtered_food['Food_ID'].values
    
    y_true = [1 if item in true_positive_items else 0 for item in user_food_matrix.columns]
    y_pred = [1 if item in recommended_items else 0 for item in user_food_matrix.columns]
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    return precision, recall, f1, accuracy

# Step 7: Main function 
def main():
    print("Masukkan preferensi makanan untuk rekomendasi:")
    selected_categories = input("Masukkan kategori makanan yang Anda suka (pisahkan dengan koma): ").strip().split(',')
    selected_categories = [cat.strip().capitalize() for cat in selected_categories]
    
    veg_non_input = input("Apakah Anda ingin makanan vegetarian atau non-vegetarian? (Masukkan 'vegetarian' atau 'non-vegetarian'): ").strip().lower()
    veg_non_input = 'veg' if veg_non_input == 'vegetarian' else 'non-veg'
    
    new_user_preferences = {
        "categories": selected_categories,
        "veg_non": veg_non_input
    }
    
    recommendations, recommendation_scores = recommend_for_new_user(
        new_user_preferences, user_food_matrix, user_similarity_df, food_df
    )
    
    if isinstance(recommendations, str):
        print(recommendations)
    else:
        print("\nRekomendasi makanan untuk Anda:")
        print(recommendations)
        
        # Perform evaluation per user input
        precision, recall, f1, accuracy = evaluate_recommendation_user_based(
            recommendations, new_user_preferences, food_df
        )
        
        print("\nEvaluasi hasil rekomendasi berdasarkan input Anda:")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(f"Accuracy: {accuracy:.2f}")

# Step 8: Execute main function
main()