import os
import utils
import gen_data
import train_models
import evaluate_model
import tensorflow as tf
import compare_cnn_cost

tf.config.run_functions_eagerly(True) #fixes colab bug about eager execution
parent_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
  print("starting the ecg cnn pipeline")

  print("\n \n \n generating data")
  gen_data.main()

  # Set the device to GPU
  with tf.device('/GPU:0'):
    # Your TensorFlow operations here
    print("GPU is being used")

  print("\n \n \n training models")
  train_models.main()

  print("\n \n \n evaluating models")
  evaluate_model.main()

  #print("compare model size and inference speeds")
  compare_cnn_cost.main()


  print("\n \n \n pipeline finished, find results in parent directory")

