# GET PYTHON FILE AND CONVERT SAVED MODEL
import os
import pickle
import argparse
import io
import zipfile
import soundfile as sf
from datasets import load_dataset, Dataset, Audio

diskdir = r'E:\\yodas\\' #external HD
dir = os.getcwd() + "\\"

def save_ds(subset):
    data_dict = []
    # Loading all objects from the file
    with open(subset + '_data.pkl', 'rb') as file:
        while True:
            try:
                data_dict.append(pickle.load(file))
            except EOFError:
                break
            except FileNotFoundError:
                break
        return data_dict
    # loaded_data = Dataset.from_list(data_dict)
    # loaded_data.save_to_disk(diskdir + subset)
    # print('Dataset saved successfully')  

def save_to_zip(data_dict, zip_filename, max_size=1 * 1024 * 1024 * 1024):
    current_size = 0
    folder_index = 1
    audio_folder_name = f"{folder_index:08d}/audio"
    text_filename = f"{folder_index:08d}/text.txt"
    combined_text_content = ""

    # Create an in-memory zip file
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
        for sample in data_dict:
            utt_id = sample['utt_id']
            
            # Prepare audio data (Assuming 'audio' is raw WAV data)
            audio_data = sample['audio'] 
            audio_wav_filename = f"{audio_folder_name}/{utt_id}.wav"

            # Calculate size of the current audio file to track total size
            audio_size = len(audio_data)

            # Check if adding this audio will exceed the max size (1GB)
            if current_size + audio_size > max_size:
                # Write the current combined text file and start a new folder
                zf.writestr(text_filename, combined_text_content)

                # Reset combined text, increment folder, and reset size counter
                folder_index += 1
                audio_folder_name = f"{folder_index:08d}/audio"
                text_filename = f"{folder_index:08d}/text.txt"
                combined_text_content = ""
                current_size = 0

            # Add text entry for this audio
            combined_text_content += f"{utt_id} {sample['text']}\n"

            # Write the audio file to the appropriate folder
            zf.writestr(audio_wav_filename, audio_data)

            # Update the current size
            current_size += audio_size

        # Write the last text file (for the remaining audios)
        if combined_text_content:
            zf.writestr(text_filename, combined_text_content)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Subset, streaming T/F"
    )
    parser.add_argument("--sub", default='es104', type=str)
    args = parser.parse_args()

    subset = args.sub

    # save_ds(subset)
    # Call the function to save the zip file
    save_to_zip(save_ds(subset), subset + '_data.zip')
