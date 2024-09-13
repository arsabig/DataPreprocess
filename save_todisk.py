import os
import pickle
import argparse
import zipfile

diskdir = r'E:\\yodas\\'  # external HD
dir = os.getcwd() + "\\"


def save_ds_in_chunks(subset, chunk_size=1000):
    # Generator function to load objects from the pickle file in chunks
    with open(subset + '_data.pkl', 'rb') as file:
        while True:
            data_chunk = []
            try:
                for _ in range(chunk_size):
                    data_chunk.append(pickle.load(file))
            except EOFError:
                break
            except FileNotFoundError:
                break
            if data_chunk:
                yield data_chunk

        # Yield any remaining chunk even if it's less than chunk_size
        if data_chunk:
            yield data_chunk


def save_to_zip_in_chunks(subset, zip_filename, chunk_size=1000, max_size=1 * 1024 * 1024 * 1024):
    current_size = 0
    folder_index = 1
    audio_folder_name = f"{folder_index:08d}/audio"
    text_filename = f"{folder_index:08d}/text.txt"
    combined_text_content = ""

    # Create a zip file
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
        for data_chunk in save_ds_in_chunks(subset, chunk_size):
            for sample in data_chunk:
                utt_id = sample['utt_id']

                # Prepare audio data (Assuming 'audio' is raw WAV data)
                audio_data = sample['audio']
                audio_wav_filename = f"{audio_folder_name}/{utt_id}.wav"

                # Calculate size of the current audio file to track total size
                audio_size = len(audio_data)

                # Check if adding this audio will exceed the max size (1GB)
                if current_size + audio_size > max_size:
                    # Write the current combined text file and reset for the next folder
                    if combined_text_content:
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

            # After each chunk, save the current text file if it's not already written
            if combined_text_content:
                zf.writestr(text_filename, combined_text_content)

        # Handle remaining files after processing full chunks
        if combined_text_content:
            folder_index += 1
            audio_folder_name = f"{folder_index:08d}/audio"
            text_filename = f"{folder_index:08d}/text.txt"
            zf.writestr(text_filename, combined_text_content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Subset, streaming T/F"
    )
    parser.add_argument("--sub", default='es104', type=str)
    args = parser.parse_args()

    subset = args.sub

    # Call the function to save the zip file in chunks
    save_to_zip_in_chunks(subset, diskdir + subset + '_data.zip')
