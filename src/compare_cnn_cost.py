import os
import time
import tensorflow as tf
from tensorflow.keras import backend as K

parent_dir = os.path.dirname(os.path.abspath(__file__))

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def measure_inference_time(model, sample_input, num_trials=100, device='/GPU:0'):
    """measure average inference time for junk input """
    times = []
    with tf.device(device):
        for _ in range(num_trials):
            start_time = time.time()
            _ = model.predict(sample_input, verbose=0)
            end_time = time.time()
            times.append(end_time - start_time)
    return sum(times) / num_trials

def get_model_size(model_path):
    """return model file size in mb"""
    size = os.path.getsize(model_path)/  (1024 * 1024)  
    return size

def compare_models():
    model_1d_path = os.path.join(parent_dir,  "1D_CNN.h5") 
    model_2d_path = os.path.join(parent_dir, "2D_CNN.h5") 
    
    #load models
    model_1d = load_model(model_1d_path)
    model_2d = load_model(model_2d_path )
    
    #create sample inputs
    sample_1d = tf.random.normal([1, 256, 1])
    sample_2d = tf.random.normal([1, 128,  128, 1])
    
    print("measuring inference time on CPU...")
    inf_time_1d_cpu = measure_inference_time(model_1d, sample_1d, device='/CPU:0')
    inf_time_2d_cpu = measure_inference_time(model_2d, sample_2d, device='/CPU:0')
    
    print("measuring inference time on GPU (if available)...")
    inf_time_1d_gpu = measure_inference_time(model_1d, sample_1d, device='/GPU:0')
    inf_time_2d_gpu= measure_inference_time(model_2d, sample_2d, device='/GPU:0')
    
    print("calculating model size...")
    model_size_1d = get_model_size(model_1d_path)  
    model_size_2d = get_model_size( model_2d_path)  

    print("counting parameters...")
    params_1d = model_1d.count_params()
    params_2d = model_2d.count_params()
    
    results = f"""
Comparison Results:
-------------------------
1D CNN - inference time (cpu): {inf_time_1d_cpu:.6f} sec 
2D CNN - inference time (cpu): {inf_time_2d_cpu:.6f} sec
1D CNN - inference time (gpu): {inf_time_1d_gpu:.6f} sec  
2D CNN - inference time (gpu): {inf_time_2d_gpu:.6f} sec

1D CNN - model size: {model_size_1d:.2f} mb 
2D CNN - model size: {model_size_2d:.2f} mb 

1D CNN - parametres: {params_1d}
2D CNN - parametres: {params_2d}
""" 
    # cast the model summary as a string
    summary_1d = []
    model_1d.summary(print_fn=lambda x: summary_1d.append(x))
    summary_2d = []
    model_2d.summary(print_fn=lambda x: summary_2d.append(x))
    
    summary_1d_str = "\n".join(summary_1d)
    summary_2d_str = "\n".join(summary_2d)

    #add summary to the output string
    results += f"""
1D CNN Model Summary:
-------------------------
{summary_1d_str}

2D CNN Model Summary:
-------------------------
{summary_2d_str}
"""
    

    #save results to a .txt file
    results_file_path = os.path.join(parent_dir, "model_comparison_results.txt")
    with open(results_file_path, 'w') as file: 
        file.write(results)

def main():
    compare_models()

if __name__ == "__main__":
    main()