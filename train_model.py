import sys

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import numpy as np

from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver

ex = Experiment(save_git_info=False)
ex.observers.append(FileStorageObserver("experiments"))
#ex.observers.append(MongoObserver(url="mongodb://mongo_user:mongo_password_IUM_2021@localhost:27017", db_name="sacred"))


@ex.config
def config():
    epochs = 10


@ex.automain
def main(epochs):
    tf.config.set_visible_devices([], 'GPU')

    # Read and split data
    train_data = pd.read_csv("crime_train.csv")
    val_data = pd.read_csv("crime_dev.csv")
    test_data = pd.read_csv("crime_test.csv")

    x_columns = ["DISTRICT", "STREET", "YEAR", "MONTH", "DAY_OF_WEEK", "HOUR", "Lat", "Long"]
    y_column = "OFFENSE_CODE_GROUP"

    x_train = train_data[x_columns]
    y_train = train_data[y_column]
    x_val = val_data[x_columns]
    y_val = val_data[y_column]
    x_test = test_data[x_columns]
    y_test = test_data[y_column]

    num_categories = len(y_train.unique())
    num_features = len(x_columns)

    # Train label encoders for categorical data
    encoder_y = LabelEncoder()
    encoder_day = LabelEncoder()
    encoder_dist = LabelEncoder()
    encoder_street = LabelEncoder()
    encoder_y.fit(y_train)
    encoder_day.fit(x_train["DAY_OF_WEEK"])
    encoder_dist.fit(x_train["DISTRICT"])
    encoder_street.fit(pd.concat([x_val["STREET"], x_test["STREET"], x_train["STREET"]], axis=0))


    # Encode train categorical data
    y_train = encoder_y.transform(y_train)
    x_train["DAY_OF_WEEK"] = encoder_day.transform(x_train["DAY_OF_WEEK"])
    x_train["DISTRICT"] = encoder_dist.transform(x_train["DISTRICT"])
    x_train["STREET"] = encoder_street.transform(x_train["STREET"])

    # Encode train categorical data
    y_val = encoder_y.transform(y_val)
    x_val["DAY_OF_WEEK"] = encoder_day.transform(x_val["DAY_OF_WEEK"])
    x_val["DISTRICT"] = encoder_dist.transform(x_val["DISTRICT"])
    x_val["STREET"] = encoder_street.transform(x_val["STREET"])

    # Encode train categorical data
    y_test = encoder_y.transform(y_test)
    x_test["DAY_OF_WEEK"] = encoder_day.transform(x_test["DAY_OF_WEEK"])
    x_test["DISTRICT"] = encoder_dist.transform(x_test["DISTRICT"])
    x_test["STREET"] = encoder_street.transform(x_test["STREET"])

    # Define model
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=num_features))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_categories, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_accuracy'])

    # Train model
    history = model.fit(x_train, y_train, epochs=int(epochs), validation_data=(x_val, y_val))

    # Make predictions
    y_pred = model.predict(x_test)
    output = [np.argmax(pred) for pred in y_pred]
    output_text = encoder_y.inverse_transform(list(output))

    # Save predictions
    data_to_save = pd.concat([test_data[x_columns], test_data[y_column]], axis = 1)
    data_to_save["PREDICTED"] = output_text
    data_to_save.to_csv("out.csv")

    # Save model
    model.save("model")
    ex.add_artifact("model/saved_model.pb")

    # Log metrics
    ex.log_scalar("loss", history.history["loss"])
    ex.log_scalar("accuracy", history.history["accuracy"])
    ex.log_scalar("val_loss", history.history["val_loss"])
    ex.log_scalar("val_accuracy", history.history["val_accuracy"])
