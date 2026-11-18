import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title='Banana quality', page_icon='🍌')
sns.set_theme(style="whitegrid")

# Name of each quality class
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


@st.cache_resource
def train_random_forest(X_train, y_train):
    # Same Random Forest as the notebook
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf


@st.cache_data
def cross_validation_scores(X_train, y_train):
    # Same search for k as the notebook (repeated cross-validation)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    scores = []
    for kk in range(1, 26):
        model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=kk))
        scores.append(cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean())
    return scores


bananas, df = load_data()
X = df[['R', 'G', 'B', 'size']]
y = df['quality']

# Same split as the notebook (same random_state), so the test set is the same
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

y_pred_knn = loaded_classifier.predict(scaler.transform(X_test))
rf = train_random_forest(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
acc_rf = accuracy_score(y_test, y_pred_rf)
labels = sorted(y.unique())


def plot_confusion(y_pred, title, cmap):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Class predicted by the model')
    ax.set_ylabel('Actual class (given by the expert)')
    ax.set_title(title)
    return fig


# =====================================================================
st.title('🍌 Banana Quality Classifier')
st.markdown(
    'This app uses **machine learning** to answer a simple question: '
    '**if I measure the color and size of a banana tier, can I guess its quality?**\n\n'
    'An expert from the Department of Agriculture in the Philippines graded 194 banana tiers by hand '
    '(a *tier*, or *hand*, is a small bunch of bananas). We used those examples to train a '
    '**k-NN** model so it learns to do the same job as the expert, and we compared it with a '
    '**Random Forest** model to see how well it performs.'
)

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

st.markdown('Go through the tabs in order to understand the project step by step.')

tab_data, tab_train, tab_results, tab_predict = st.tabs(
    ['1. The data', '2. How it was trained', '3. Results', '4. Try the model'])

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

    st.subheader('An important detail: each banana appears 6 times')
    st.markdown(
        f'The original file has **{len(bananas):,} rows**, but only **{df.shape[0]} banana tiers**. '
        'Each tier was photographed from 6 sides (front, back, left, right, top and bottom).\n\n'
        'If we used the rows as they are, photos of the same banana could end up some in training '
        'and others in testing. The model would be tested on a banana it has already seen, and it would look '
        'better than it really is. That is why we **average the color of the 6 photos** and keep one row per banana.'
    )
    st.dataframe(df.head(), hide_index=True)
    st.caption('This is how the data looks after averaging: one row per banana tier.')

    st.subheader('How many bananas are there in each class?')
    fig, ax = plt.subplots()
    sns.countplot(x='quality', hue='quality', data=df, palette='viridis', legend=False, ax=ax)
    ax.set_xlabel('Quality class (1 = extra, 4 = reject)')
    ax.set_ylabel('Number of banana tiers')
    st.pyplot(fig)
    st.markdown(
        f'In the graph, the classes are not balanced. Class 1 has {counts[1]} tiers, '
        f'but class 3 only has {counts[3]}. With fewer examples, it is harder for the model to learn '
        'to recognize class 3, and we will see that in the results.'
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
        'Classes 1 and 3 are close in size, which is why the model sometimes confuses them.'
    )

# ---------------------------------------------------------------------
with tab_train:
    st.subheader('Step 1: split the data')
    st.markdown(
        f'We used {len(X_train)} tiers (80%) to train the models and kept {len(X_test)} tiers (20%) '
        'to test them. The models never see the test tiers while learning.'
    )

    st.subheader('Step 2: the k-NN model')
    st.markdown(
        f'To classify a new banana, k-NN looks for the {k} most similar bananas in the training data '
        'and picks the most common class among them. The data is scaled first, so size does not '
        'dominate the colors.'
    )

    st.subheader(f'Step 3: why k = {k}?')
    st.markdown('We tested k from 1 to 25 using cross-validation on the training data only.')
    with st.spinner('Running cross-validation...'):
        cv_scores = cross_validation_scores(X_train, y_train)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, 26), cv_scores, marker='o')
    ax.axvline(k, color='red', linestyle='--', label=f'Chosen k = {k}')
    ax.set_xlabel('Number of neighbors (k)')
    ax.set_ylabel('Cross-validated accuracy')
    ax.legend()
    st.pyplot(fig)
    best = int(np.argmax(cv_scores)) + 1
    st.markdown(
        f'In the graph, the best result was with k = {best} ({cv_scores[best - 1]:.1%}), but k = {k} '
        f'was practically tied ({cv_scores[k - 1]:.1%}). We chose k = {k} because looking at more neighbors '
        'makes the model more stable.'
    )

    st.subheader('Step 4: why Random Forest?')
    st.markdown(
        'We used it to compare with k-NN. It works differently (100 decision trees that vote), '
        'it does not need scaled data, and it shows which variables matter most.'
    )

# ---------------------------------------------------------------------
with tab_results:
    st.header('How well do the models work?')
    st.markdown(
        f'Here we evaluate both models on the **{len(X_test)} test tiers**, the ones they did not see during training. '
        'We measure the **accuracy**: the percentage of bananas the model classified correctly.'
    )
    col1, col2 = st.columns(2)
    col1.metric(f'k-NN (k = {k})', f'{acc_knn:.1%}',
                help=f'Got {int(round(acc_knn * len(X_test)))} of {len(X_test)} tiers right')
    col2.metric('Random Forest', f'{acc_rf:.1%}',
                help=f'Got {int(round(acc_rf * len(X_test)))} of {len(X_test)} tiers right')

    st.subheader('Confusion matrices: where does each model make mistakes?')
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_confusion(y_pred_knn, f'k-NN (k = {k})', 'Reds'))
    with col2:
        st.pyplot(plot_confusion(y_pred_rf, 'Random Forest', 'Greens'))
    st.markdown(
        'In the graphs, the diagonal shows the correct predictions and everything outside it is a mistake. '
        'Both models get all class 1 bananas right, but they confuse some class 3 bananas with class 1 '
        'and some class 2 bananas with class 4. Random Forest makes fewer mistakes than k-NN.'
    )

    st.subheader('Which variables matter most for the decision?')
    feature_importances = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})
    feature_importances = feature_importances.sort_values(by='importance', ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x='importance', y='feature', hue='feature', data=feature_importances,
                palette='viridis', legend=False, ax=ax)
    ax.set_xlabel('Importance')
    ax.set_ylabel('Variable')
    ax.set_title('Feature importance for banana quality classification')
    st.pyplot(fig)
    imp = feature_importances.set_index('feature')['importance']
    st.markdown(
        f'In the graph, finger length is by far the most important variable ({imp["size"]:.0%}), '
        f'while each color only contributes between {imp[["R", "G", "B"]].min():.0%} and '
        f'{imp[["R", "G", "B"]].max():.0%}. This matches the boxplot: size separates the classes '
        'much better than color.'
    )

    st.subheader('Limitations')
    st.markdown(
        '- The test set is small, so the percentages could change with a different split.\n'
        '- The classes do not have the same number of bananas, and class 3 has few.\n'
        '- The classes were assigned by a person, so part of the "error" may come from the expert\'s '
        'grading being somewhat subjective.'
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

    # Feature table for prediction (same column names as in training)
    features = pd.DataFrame({'R': [r], 'G': [g], 'B': [b], 'size': [size]})

    if st.button('Predict quality'):
        # k-NN needs the same scaling used during training
        features_scaled = scaler.transform(features)
        prediction = loaded_classifier.predict(features_scaled)[0]
        votes = loaded_classifier.predict_proba(features_scaled)[0]

        st.success(f'Result: **{CLASS_NAMES[int(prediction)]}**')

        votes_text = ', '.join(
            f'{int(round(v * k))} from class {int(c)}'
            for c, v in zip(loaded_classifier.classes_, votes) if v > 0)
        st.markdown(f'Of the {k} most similar bananas in the training data, there were {votes_text}.')

        rf_prediction = rf.predict(features)[0]
        if rf_prediction == prediction:
            st.markdown('Random Forest also says it belongs to this class.')
        else:
            st.markdown(f'Random Forest disagrees: it would say **{CLASS_NAMES[int(rf_prediction)]}**.')

# streamlit run banana_inference.py
