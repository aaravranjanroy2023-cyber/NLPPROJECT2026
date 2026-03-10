import pandas as pd
import numpy as np
import textstat

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC




def load_dataset(file_path, text_col, score_col, time_col):

    df = pd.read_csv(file_path)

    df = df.rename(columns={
        text_col: "body",
        score_col: "score",
        time_col: "created_utc"
    })

    return df




print("Loading dataset...")

df = load_dataset(
    "the-reddit-dataset-dataset-comments.csv",
    text_col="body",
    score_col="score",
    time_col="created_utc"
)




df = df.dropna(subset=['body'])

df['body'] = df['body'].astype(str)

df = df[df['body'] != '[deleted]']

df['created_utc'] = pd.to_datetime(df['created_utc'], errors='coerce')

df = df.dropna(subset=['created_utc'])



df['hour'] = df['created_utc'].dt.hour
df['dayofweek'] = df['created_utc'].dt.dayofweek

df['word_count'] = df['body'].apply(lambda x: len(x.split()))

df['char_length'] = df['body'].apply(len)


def ttr(text):
    words = text.split()
    return len(set(words)) / len(words) if len(words) > 0 else 0


df['ttr'] = df['body'].apply(ttr)

df['num_exclamation'] = df['body'].str.count('!')
df['num_question'] = df['body'].str.count(r'\?')


def capital_ratio(text):
    return sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0


df['capital_ratio'] = df['body'].apply(capital_ratio)

df['readability'] = df['body'].apply(
    lambda x: textstat.flesch_reading_ease(x)
)

df['avg_word_len'] = df['body'].apply(
    lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0
)




threshold = df["score"].median()

df["low_engagement"] = (df["score"] <= threshold).astype(int)




numeric_features = [
    'sentiment',
    'word_count',
    'char_length',
    'ttr',
    'hour',
    'dayofweek',
    'num_exclamation',
    'num_question',
    'capital_ratio',
    'readability',
    'avg_word_len'
]

text_feature = "body"

df = df.dropna(subset=numeric_features)

X = df[numeric_features + [text_feature]]
y = df["low_engagement"]




numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

text_transformer = TfidfVectorizer(
    max_features=1500,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("text", text_transformer, text_feature)
    ]
)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)




models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=60,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    ),

    "SVM": LinearSVC(
        class_weight="balanced"
    )
}




for name, clf in models.items():

    print("\n=============================")
    print("TRAINING MODEL:", name)
    print("=============================")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    pipeline.fit(X_train, y_train)

    print(name, "training finished")

    pred = pipeline.predict(X_test)

    print(classification_report(y_test, pred))

    print(df.columns)
