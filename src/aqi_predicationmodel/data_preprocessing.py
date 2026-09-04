from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


def preprocessing(df):

    df = df.drop_duplicates()

    X = df.drop(columns=['AH','Date','Time'])
    y = df['AH']

    categorical_data = X.select_dtypes(include='object').columns
    numerical_data = X.select_dtypes(exclude='object').columns

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=1
    )

    numerical_pipeline = Pipeline(steps=[('Imputer', SimpleImputer(strategy='median')),
                                         ('Scaler', RobustScaler())])

    categorical_pipeline = Pipeline(steps=[('Imputer', SimpleImputer(strategy='most_frequent')),
                                           ('Encoder', OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False))])

    transformer = ColumnTransformer(transformers=[('num', numerical_pipeline, numerical_data),
                                                  ('cat', categorical_pipeline, categorical_data)])

    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)

    return X_train, X_test, y_train, y_test, transformer