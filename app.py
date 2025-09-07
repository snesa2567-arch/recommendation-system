import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast

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
top_n = st.slider("How many matches to show?", 1, 20, 10)


if selected_color:

   
    input_row = df[df[color_col] == selected_color].iloc[0]
    rgb_col = [col for col in df.columns if "RGB" in col][0]  
    input_rgb = np.array(ast.literal_eval(input_row[rgb_col])).reshape(1, -1)

    results = []
    rgb_col_match = [col for col in match_df.columns if "RGB" in col][0]

    for _, row in match_df.iterrows():
        row_rgb = np.array(ast.literal_eval(row[rgb_col_match])).reshape(1, -1)
        sim = cosine_similarity(input_rgb, row_rgb)[0][0]
        results.append({
            "Color": row[match_col],
            "Similarity": sim,
            "Accessories": row.get("Accessories", None),
            "Popularity": row.get("Popularity Score", 0)
        })

    
    results = sorted(
        results,
        key=lambda x: (x["Similarity"] * 0.7 + x["Popularity"] * 0.3),
        reverse=True
    )

    
    st.subheader("🎨 Recommended Matches")
    for res in results[:top_n]:
        st.markdown(
            f"✅ **Match:** `{res['Color']}`  \n"
            f"📊 **Similarity:** {res['Similarity']:.2f} | ⭐ **Popularity:** {res['Popularity']:.1f}"
        )
        if show_accessories and res["Accessories"]:
            st.markdown(f"🧢 *Accessories:* {res['Accessories']}")
        if res["Image"]:
            st.image(res["Image"], width=150)
