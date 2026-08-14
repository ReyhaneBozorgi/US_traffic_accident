from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.compose import ColumnTransformer

def build_preprocess(df,target):

    numeric_features = df.drop(target,axis=1).select_dtypes(include=['int','float']).columns
    categorical_features = df.drop(target,axis=1).select_dtypes(include=['object']).columns

    numeric_scaler = StandardScaler()
    categorical_scaler = OneHotEncoder()

    preprocessor = ColumnTransformer([
        ('num',numeric_scaler,numeric_features),
        ('cat',categorical_scaler,categorical_features)
    ])

    return preprocessor