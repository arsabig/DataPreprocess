# GET PYTHON FILE AND CONVERT SAVED MODEL
import os
import pickle
import argparse
from datasets import load_dataset, Dataset, Audio

savedir = r'E:\\yodas\\' #external HD
dir = os.getcwd() + "\\"

def save_ds(subset):
    metadatas = []
    # Loading all objects from the file
    with open(subset + '_data.pkl', 'rb') as file:
        while True:
            try:
                metadatas.append(pickle.load(file))
            except EOFError:
                break
            except FileNotFoundError:
                break
    loaded_data = Dataset.from_list(metadatas)
    loaded_data.save_to_disk(savedir + subset)
    print('Dataset saved successfully')  


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Subset, streaming T/F"
    )
    parser.add_argument("--sub", default='es104', type=str)
    args = parser.parse_args()

    subset = args.sub

    save_ds(subset)
