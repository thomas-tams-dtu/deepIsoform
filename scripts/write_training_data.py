import os

def write_training_data(file_path, network_name, network_size, latent_features, learning_rate, weight_decay, patience, training_runs, train_loss, eval_loss):
    # Check if file exists
    print(os.path.exists(file_path))
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            header_list = ['network_name', 'size', 'latent_features', 'learning_rate', 'weight_decay', 'patience', 'training_runs', 'train_loss', 'eval_loss']
            header_string = '\t'.join(map(str, header_list)) + '\n'
            file.write(header_string)
    
    # Read the existing content
    with open(file_path, 'r') as file:
        existing_content = file.read()

    # Add a line to the existing content
    training_data_list = [network_name, network_size, latent_features, learning_rate, weight_decay, patience, training_runs, train_loss, eval_loss]
    line_to_add = '\t'.join(map(str, training_data_list)) + '\n'
    new_content = existing_content + line_to_add

    # Write the new content back to the file
    with open(file_path, 'w') as file:
        file.write(new_content)


"""
NETWORK_SIZE = 'small'
LATENT_FEATURES = 16
BATCH_SIZE = 500
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 100
PATIENCE = 6
MODEL_NAME = f'PCA_DENSE_l{LATENT_FEATURES}_lr{LEARNING_RATE}_e{NUM_EPOCHS}_wd{WEIGHT_DECAY}_p{PATIENCE}'
IPCA = f'/zhome/99/d/155947/DeeplearningProject/deepIsoform/models/ipca_model_n{LATENT_FEATURES}.pkl'

epoch = 36

PARAMETER_SAVE_PATH = f'/zhome/99/d/155947/DeeplearningProject/deepIsoform/data/training_meta_data/train_data_{NETWORK_SIZE}.tsv'
write_training_data(file_path=PARAMETER_SAVE_PATH,
                    network_name=MODEL_NAME,
                    network_size=NETWORK_SIZE,
                    latent_features=LATENT_FEATURES,
                    learning_rate=LEARNING_RATE,
                    weight_decay=WEIGHT_DECAY,
                    patience=PATIENCE,
                    training_runs=epoch,
                    train_loss= [4432, 4324, 3252, 2411, 1243, 590],
                    eval_loss=  [5430, 4358, 8659, 9023, 3849, 1009])
"""
