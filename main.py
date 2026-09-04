from src.aqi_predicationmodel.data_ingestion import load_data
from src.aqi_predicationmodel.data_preprocessing import preprocessing
from src.aqi_predicationmodel.model_build import model
def main():
    df=load_data()
    print(df.shape)

    X_train,X_test,y_train,y_test,transformer= preprocessing(df)
    print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

    y_pred,r2score=model(X_train,X_test,y_train,y_test)



if __name__ == "__main__":
    
    main()
