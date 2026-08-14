from sklearn.model_selection import train_test_split

def train_model(df,targrt,pipe):

    X = df.drop(targrt,axis = 1)
    y = df[targrt]

    X_train , X_test , y_train , y_test = train_test_split(
        X,y,test_size=0.2,random_state=42)

    pipe.fit(X_train,y_train)

    return pipe, X_test , y_test , X_train , y_train
