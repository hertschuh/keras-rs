from keras.utils import Config

# === Training Hyperparameters ===
training_config = Config()
training_config.learning_rate = 0.0034
training_config.global_batch_size = 128
# Set num_steps in the main config file instead of num_epochs, because we are
# using a Python generator.
# training_config.num_epochs = 1
