# Directory paths
model_folder = 'recovery/DeepFTSrc/checkpoints/'
data_folder = 'recovery/DeepFTSrc/data/'
plot_folder = 'recovery/DeepFTSrc/plots'
data_filename = 'time_series.npy'
schedule_filename = 'schedule_series.npy'

# Hyperparameters
num_epochs = 35
PERCENTILES = 98
PROTO_DIM = 2
PROTO_UPDATE_FACTOR = 0.2
PROTO_UPDATE_MIN = 0.02
PROTO_FACTOR_DECAY = 0.995
K = 1