import tensorflow_datasets as tfds
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

# We can just load the dataset builder to see the feature dict
builder = tfds.builder("movi_e", data_dir="gs://kubric-public/tfds")
info = builder.info
print("Feature structure:")
print(info.features)
