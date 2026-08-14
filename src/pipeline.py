from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def build_pipeline(preprocessor):

    pipe = Pipeline([
        ('scaler',preprocessor),
        ('model',LogisticRegression(
            max_iter=10000
        ))
    ])

    pipe1 = Pipeline([
        ('scaler',preprocessor),
        ('model',XGBClassifier(
            n_estimators = 200
        ))
    ])

    return pipe

