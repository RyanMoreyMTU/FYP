import os
import tensorflow as tf
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
print("GPU Available:", tf.config.list_physical_devices('GPU'))



import tensorflow as tf