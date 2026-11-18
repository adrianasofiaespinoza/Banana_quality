# Streamlit web app for the "Banana Quality Classification" notebook (k-NN)
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title='Banana quality', page_icon='🍌')
sns.set_theme(style="whitegrid")

CLASS_NAMES = {
    1: 'Class 1 – Extra (the best)',
    2: 'Class 2 – Intermediate',
    3: 'Class 3 – Intermediate',
    4: 'Class 4 – Reject (the worst)',
}

saved = joblib.load('knn_banana_model.pkl')
scaler = saved['scaler']
loaded_classifier = saved['model']
k = loaded_classifier.n_neighbors


@st.cache_data
def load_data():
    # Same preprocessing as the notebook: one row per banana tier
    bananas = pd.read_csv('banana_features.csv')
    df = (bananas.groupby('SampleID')
                 .agg(R=('R', 'mean'), G=('G', 'mean'), B=('B', 'mean'),
                      size=('size', 'first'), quality=('class', 'first'))
                 .reset_index(drop=True))
    return bananas, df


bananas, df = load_data()
X = df[['R', 'G', 'B', 'size']]
y = df['quality']

# =====================================================================
st.title('🍌 Banana Quality Classifier')
st.markdown(
    'This app uses **machine learning** to answer a simple question: '
    '**if I measure the color and size of a banana tier, can I guess its quality?**\n\n'
    'An expert from the Department of Agriculture in the Philippines graded 194 banana tiers by hand '
    '(a *tier*, or *hand*, is a small bunch of bananas). We used those examples to train a '
    '**k-NN** model so it learns to do the same job as the expert.'
)
st.markdown('📂 Source code: [GitHub repository](https://github.com/adrianasofiaespinoza/Banana_quality)')

st.subheader('The 4 quality classes')
counts = y.value_counts().sort_index()
classes_table = pd.DataFrame({
    'Class': [CLASS_NAMES[c] for c in counts.index],
    'Tiers in the data': counts.values,
})
st.dataframe(classes_table, hide_index=True)
st.markdown(
    'Class 1 is the best commercial quality and class 4 is rejected. '
    'Classes 2 and 3 are the intermediate grades.'
)


tab_data, tab_predict = st.tabs(['1. The data', '2. Try the model'])

# ---------------------------------------------------------------------
with tab_data:
    st.header('What was measured for each banana?')
    st.markdown(
        'For each banana tier we have 4 measurements. These are what the model uses to decide:\n\n'
        '- **R, G and B**: the average color of the photo, in red, green and blue. '
        'Each value goes from 0 to 255. A greener banana has a higher G compared to R.\n'
        '- **size**: the length of the top middle finger, measured by hand in millimeters.\n\n'
        'And the **class** (1 to 4) given by the expert, which is what we want to predict.'
    )

    st.subheader('How many bananas are there in each class?')
    fig, ax = plt.subplots()
    sns.countplot(x='quality', hue='quality', data=df, palette='viridis', legend=False, ax=ax)
    ax.set_xlabel('Quality class (1 = extra, 4 = reject)')
    ax.set_ylabel('Number of banana tiers')
    st.pyplot(fig)
    st.markdown(
        f'In the graph, the classes are not balanced. Class 1 has {counts[1]} tiers, '
        f'but class 3 only has {counts[3]}. With fewer examples, it is harder for the model to learn '
        'to recognize class 3.'
    )

    st.subheader('Does the size change with the class?')
    fig, ax = plt.subplots()
    sns.boxplot(x='quality', y='size', hue='quality', data=df, palette='viridis', legend=False, ax=ax)
    ax.set_xlabel('Quality class (1 = extra, 4 = reject)')
    ax.set_ylabel('Finger length (mm)')
    st.pyplot(fig)
    medians = df.groupby('quality')['size'].median()
    st.markdown(
        f'In the graph, size separates the classes very well: the median is about {medians[3]:.0f} mm in '
        f'class 3, {medians[1]:.0f} mm in class 1, {medians[2]:.0f} mm in class 2 and {medians[4]:.0f} mm in '
        'class 4. Class 3 has the longest fingers, not class 1, so quality is not just "bigger is better". '
        'Classes 1 and 3 are close in size, so they are the easiest to confuse.'
    )

# ---------------------------------------------------------------------
with tab_predict:
    st.header('Try the model with a new banana')
    st.markdown(
        'Enter the measurements of a banana tier and the k-NN model will tell you its quality class. '
        'The default values belong to an invented banana that is not in the data.'
    )

    r = st.number_input('Mean red value (R)', min_value=0.0, max_value=255.0, value=85.0,
                        help=f'In the data it goes from {X["R"].min():.0f} to {X["R"].max():.0f}')
    g = st.number_input('Mean green value (G)', min_value=0.0, max_value=255.0, value=108.0,
                        help=f'In the data it goes from {X["G"].min():.0f} to {X["G"].max():.0f}')
    b = st.number_input('Mean blue value (B)', min_value=0.0, max_value=255.0, value=48.0,
                        help=f'In the data it goes from {X["B"].min():.0f} to {X["B"].max():.0f}')
    size = st.number_input('Finger length (mm)', min_value=0.0, max_value=300.0, value=150.0,
                           help=f'In the data it goes from {X["size"].min():.0f} to {X["size"].max():.0f} mm')

    features = pd.DataFrame({'R': [r], 'G': [g], 'B': [b], 'size': [size]})

    if st.button('Predict quality'):
        features_scaled = scaler.transform(features)
        prediction = loaded_classifier.predict(features_scaled)[0]
        votes = loaded_classifier.predict_proba(features_scaled)[0]

        st.success(f'Result: **{CLASS_NAMES[int(prediction)]}**')

        votes_text = ', '.join(
            f'{int(round(v * k))} from class {int(c)}'
            for c, v in zip(loaded_classifier.classes_, votes) if v > 0)
        st.markdown(f'Of the {k} most similar bananas in the training data, there were {votes_text}.')


# streamlit run banana_inference.py
