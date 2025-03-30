from model.randomforest import RandomForest
from data_model import Data


def model_predict(data, df, name):
    # Initialize RandomForest model
    model = RandomForest(
        model_name=name,
        embeddings=data.get_embeddings(),
        y=data.get_type()[Config.CLASS_COL]  # Using CLASS_COL from Config
    )

    # Train the model
    model.train(data)

    # Make predictions
    model.predict(data.get_X_test())

    return model


def model_evaluate(model, data):
    model.print_results(data)