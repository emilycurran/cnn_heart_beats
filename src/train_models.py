import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint


import paper_model 
import paper_model_1d 
import utils

# set as a global variable
parent_dir = os.path.dirname(os.path.abspath(__file__))

def save_accuracy(history, model_name="model", file_name="accuracy.txt"):
    with open(file_name, 'a') as f:
        if os.stat(file_name).st_size == 0:
            f.write("epoch, training accuracy, validation accuracy\n")
        
        for epoch in range(len(history.history["accuracy"])):
            train_accuracy = history.history["accuracy"][epoch]  # training accuracy
            val_accuracy = history.history["val_accuracy"][epoch]  # validation accuracy
          
            f.write(f"{epoch+1}, {train_accuracy:.4f}, {val_accuracy:.4f}\n")

def run_experiment(database_path, num_epochs, batch_size, train_split=0.8):
    # Get the data 
    train_imgs, val_imgs, train_series, val_series, train_labels, val_labels, label_map = utils.split_data(database_path) 
    
    #define the learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = 0.0001,  
        decay_steps=1000,  
        decay_rate=0.95,
        staircase=False  
    )

    # define the stopping scheduler
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", 
        min_delta=0.001, 
        patience=20, 
        verbose=1, 
        restore_best_weights=True 
    )

    # saves best model
    checkpoint1d = ModelCheckpoint(
        os.path.join(parent_dir, "1D_CNN.h5"),  # Saves the best model file
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    checkpoint2d = ModelCheckpoint(
        os.path.join(parent_dir, "2D_CNN.h5"),  # Saves the best model file
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    # ---------------------2D----------------------
    optimizer2d = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model_2d = paper_model.build_model(input_shape=(128, 128, 1))
    model_2d.compile(optimizer=optimizer2d, loss="sparse_categorical_crossentropy", metrics=["accuracy"], jit_compile=True)

    # train the model and save accuracy
    history_2d = model_2d.fit(
        train_imgs, 
        train_labels, 
        validation_data=(val_imgs, val_labels), 
        epochs=num_epochs, 
        batch_size=batch_size, 
        callbacks=[early_stopping, checkpoint2d]
    )

    # save model and acc to file
    save_accuracy(history_2d, model_name="2D_CNN", file_name="2d_training_accuracy.txt")
    #save plots
    plt.figure(figsize=(8, 6))
    plt.plot(history_2d.history["loss"], label="Training Loss")
    plt.plot(history_2d.history["val_loss"], label="Validation Loss")
    plt.title("2D CNN Loss vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(parent_dir, "2D_CNN_Loss.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(history_2d.history["accuracy"], label="Training Accuracy")
    plt.plot(history_2d.history["val_accuracy"], label="Validation Accuracy")
    plt.title("2D CNN Accuracy vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(parent_dir, "2D_CNN_Accuracy.png"))
    plt.close()    

    # -----------------------1D--------------------------
    optimizer1d = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model_1d = paper_model_1d.build_model(input_shape=(256, 1))
    model_1d.compile(optimizer=optimizer1d, loss="sparse_categorical_crossentropy", metrics=["accuracy"], jit_compile=True)

    #train the model and save accuracy
    history_1d = model_1d.fit(
        train_series, 
        train_labels, 
        validation_data=(val_series, val_labels), 
        epochs=num_epochs, 
        batch_size=batch_size, 
        callbacks=[early_stopping, checkpoint1d]
    )

    #save the training accuracy to a file
    save_accuracy(history_1d, model_name="1D_CNN", file_name="1d_training_accuracy.txt")


    #save training results (plots)
    plt.figure(figsize=(8, 6))
    plt.plot(history_1d.history["loss"], label="Training Loss") 
    plt.plot(history_1d.history["val_loss"], label="Validation Loss")
    plt.title("1D CNN Loss vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(parent_dir, "1D_CNN_Loss.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(history_1d.history["accuracy"], label="Training Accuracy") 
    plt.plot(history_1d.history["val_accuracy"], label="Validation Accuracy")
    plt.title("1D CNN Accuracy vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()  
    plt.grid(True) 
    plt.savefig(os.path.join(parent_dir, "1D_CNN_Accuracy.png"))
    plt.close() 

    return

def main():
    database_name = "ecg_analysis.db"
    database_path = os.path.join(parent_dir, database_name)
    run_experiment(database_path, num_epochs=100, batch_size=256)

if __name__ == "__main__":
    main()
