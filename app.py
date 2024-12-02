from flask import Flask, request, render_template
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load datasets
food_df = pd.read_csv('food.csv')
ratings_df = pd.read_csv('ratings.csv')

# Data preprocessing
ratings_df = ratings_df.dropna()
food_df = food_df.dropna()

# Merge datasets
merged_df = pd.merge(ratings_df, food_df, on='Food_ID', how='inner')

# Create user-food matrix
user_food_matrix = merged_df.pivot_table(index='User_ID', columns='Food_ID', values='Rating', fill_value=0)

# Calculate user similarity
user_similarity = cosine_similarity(user_food_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_food_matrix.index, columns=user_food_matrix.index)


def recommend_for_new_user(new_user_preferences, user_food_matrix, user_similarity_df, food_df, top_n=5):
    # Filter makanan berdasarkan kategori dan jenis (preferences pengguna baru)
    filtered_food = food_df[
        (food_df['C_Type'].str.strip().str.capitalize().isin(new_user_preferences['categories'])) &
        (food_df['Veg_Non'].str.strip().str.lower() == new_user_preferences['veg_non'])
    ]

    if filtered_food.empty:
        return "Tidak ada makanan yang sesuai dengan preferensi Anda."

    filtered_food_ids = filtered_food['Food_ID'].values
    valid_food_ids = [fid for fid in filtered_food_ids if fid in user_food_matrix.columns]

    if not valid_food_ids:
        return "Tidak ada makanan untuk direkomendasikan berdasarkan preferensi Anda."

    average_user_preferences = user_food_matrix.mean(axis=1)
    most_similar_user = average_user_preferences.idxmax()
    similar_user_ratings = user_food_matrix.loc[most_similar_user]
    similar_user_ratings = similar_user_ratings[valid_food_ids]
    recommendations = similar_user_ratings.sort_values(ascending=False).head(top_n)

    if recommendations.empty:
        return "Tidak ada makanan untuk direkomendasikan berdasarkan pengguna serupa."

    # Mengambil data rekomendasi dan menambah kategori dan tipe
    recommended_food = food_df[food_df['Food_ID'].isin(recommendations.index)]
    recommended_food = recommended_food[['Food_ID', 'Name', 'C_Type', 'Veg_Non']]
    return recommended_food.to_dict(orient='records')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def recommend():
    # Ambil data dari form
    selected_categories = request.form.getlist('foodPreferences')  # List dari checkbox
    veg_non_input = request.form.get('veg_non')  # Dropdown pilihan vegetarian/non-vegetarian

    # Debugging untuk cek data yang masuk
    print("Selected Categories:", selected_categories)
    print("Vegetarian/Non-Vegetarian:", veg_non_input)
    

    # Jika tidak ada input, tampilkan pesan error
    if not selected_categories or not veg_non_input:
        return render_template('recom.html', error="Input tidak lengkap. Mohon pilih preferensi makanan.")

    # Siapkan preferences
    new_user_preferences = {
        "categories": selected_categories,
        "veg_non": veg_non_input
    }

    # Buat rekomendasi
    recommendations = recommend_for_new_user(new_user_preferences, user_food_matrix, user_similarity_df, food_df)

    if isinstance(recommendations, str):  # Jika error message
        return render_template('recom.html', error=recommendations)
    else:
        return render_template('recom.html', recommendations=recommendations)


if __name__ == '__main__':
    app.run(debug=True)
