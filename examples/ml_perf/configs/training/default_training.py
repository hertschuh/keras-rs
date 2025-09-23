from keras.utils import Config

# === Training Hyperparameters ===
training_config = Config()
training_config.learning_rate = 0.005
training_config.global_batch_size = 32
training_config.num_epochs = 1
