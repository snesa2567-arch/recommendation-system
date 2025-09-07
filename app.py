import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_data
def load_datasets():
    male_tops = pd.read_csv("male_tops.csv")
    male_bottoms = pd.read_csv("male_bottoms.csv")
    female_tops = pd.read_csv("female_tops.csv")
    female_bottoms = pd.read_csv("female_bottoms.csv")
    return male_tops, male_bottoms, female_tops, female_bottoms

male_tops, male_bottoms, female_tops, female_bottoms = load_datasets()


st.title("👕 Outfit Recommendation System")

gender = st.selectbox("Select your gender:", ["Male", "Female"])

if gender == "Male":
    choice = st.radio("What do you want to start with?", ["Shirt (Top)", "Pant (Bottom)"])
    if choice == "Shirt (Top)":
        df = male_tops
        match_df = male_bottoms
        color_col = "Top Color"
        match_col = "Bottom Color"
    else:
        df = male_bottoms
        match_df = male_tops
        color_col = "Bottom Color"
        match_col = "Top Color"

else:  
    choice = st.radio("What do you want to start with?", ["Top", "Bottom"])
    if choice == "Top":
        df = female_tops
        match_df = female_bottoms
        color_col = "Top Color"
        match_col = "Bottom Color"
    else:
        df = female_bottoms
        match_df = female_tops
        color_col = "Bottom Color"
        match_col = "Top Color"


selected_color = st.selectbox(f"Choose your {color_col.lower()}:", df[color_col].unique())

show_accessories = st.checkbox("Show accessories?")

if selected_color:

    input_row = df[df[color_col] == selected_color].iloc[0]
    rgb_col = [col for col in df.columns if "RGB" in col][0]  # detect RGB column
    input_rgb = np.array(eval(input_row[rgb_col])).reshape(1, -1)

    results = []
    for _, row in match_df.iterrows():
        rgb_col_match = [col for col in match_df.columns if "RGB" in col][0]
        row_rgb = np.array(eval(row[rgb_col_match])).reshape(1, -1)
        sim = cosine_similarity(input_rgb, row_rgb)[0][0]
        results.append({
            "Color": row[match_col],
            "Similarity": sim,
            "Accessories": row.get("Accessories", None),
            "Popularity": row.get("Popularity Score", 0)
        })

    
    results = sorted(results, key=lambda x: x["Popularity"], reverse=True)

    st.subheader("Recommended Matches 🎨")
    for res in results[:10]:
        st.write(f"✅ Match: {res['Color']} | Popularity: {res['Popularity']}")
        if show_accessories and res["Accessories"]:
            st.write(f"   Accessories: {res['Accessories']}")
