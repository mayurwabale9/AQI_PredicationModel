from flaml.automl import AutoML
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

def model(X_train,X_test,y_train,y_test):
    model=RandomForestRegressor(n_estimators=100)

    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)

    r2score=r2_score(y_pred,y_test)

    print("the r2score is ",r2score)
    
    # Cross-validation
   
    kf = KFold(n_splits=2,shuffle=True,)

    cv_scores = cross_val_score(model,X_train,y_train,cv=kf,scoring="r2")
    cv_mean = cv_scores.mean()

    print("Cross-validation scores:", cv_scores)
    print("Mean cross-validation R2 score:", cv_mean)

    return y_pred,r2score