from src.preprocces import build_preprocess
from src.pipeline import build_pipeline
from src.evaluate import evaluate_model , check_overfit
from src.train import train_model

from config.setting import(
    DATA_PATH,
    TARGET
)

import joblib
import pandas as pd

df = pd.read_csv(DATA_PATH)

preprocessor = build_preprocess(df,TARGET)

pipe = build_pipeline(preprocessor)

model , X_test , y_test , X_train , y_train =train_model(df,TARGET,pipe)

joblib.dump(model,'models/Logistic_US_Accidents.pkl')

evaluate_model(model,X_test,y_test)

check_overfit(X_train,y_train,y_test,X_test,model)

