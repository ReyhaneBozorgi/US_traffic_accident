from sklearn.metrics import recall_score,accuracy_score

def evaluate_model(model , X_test , y_test ):

    pred = model.predict(X_test)

    print('accuracy total : ',accuracy_score(y_test,pred))
    print('recall : ',recall_score(y_test,pred))


def check_overfit(X_train,y_train, y_test,X_test,model):

    pred_train = model.predict(X_train)
    print('train accuracy : ',accuracy_score(y_train,pred_train))

    pred_test = model.predict(X_test)
    print('test accuracy : ',accuracy_score(y_test,pred_test))

