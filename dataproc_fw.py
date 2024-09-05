import os
import torch
import gc
import sys
import string
import logging
import pickle
import argparse
import soundfile as sf
import io
from pathlib import Path
from multiprocessing import Pool, cpu_count, get_context
from datasets import load_dataset, Dataset, Audio
from tqdm import tqdm  # For progress bar
from faster_whisper import WhisperModel
from joblib import Parallel, delayed
from logging import getLogger

LOG = getLogger(__name__)
LOG.setLevel(logging.WARNING)
MODEL_MEMORY = 5000
# from utils import get_total_gpu_memory
       # Execute in parallel
memory = torch.cuda.mem_get_info(device='cuda:0')[0]//1024 ** 2
n_jobs = min(
    max(
        memory
        // MODEL_MEMORY
        if memory is not None
        else 1,
        1,
    ),
    cpu_count(),
) 
# LOG.info(f"n_jobs automatically set to {n_jobs}, memory: {memory} MiB")

# n_jobs = min(len(configs) // 16 + 1, n_jobs)

limit_rows = 100 # limit for first x rows in the dataset
languages = [
    'af', 'am', 'ar', 'as', 'az', 'ba', 'be', 'bg', 'bn', 'bo', 'br', 'bs', 'ca', 'cs', 'cy', 'da', 'de', 'el', 'en', 
    'es', 'et', 'eu', 'fa', 'fi', 'fo', 'fr', 'gl', 'gu', 'ha', 'haw', 'he', 'hi', 'hr', 'ht', 'hu', 'hy', 'id', 'is', 
    'it', 'ja', 'jw', 'ka', 'kk', 'km', 'kn', 'ko', 'la', 'lb', 'ln', 'lo', 'lt', 'lv', 'mg', 'mi', 'mk', 'ml', 'mn', 
    'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'nn', 'no', 'oc', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'sa', 'sd', 'si', 'sk', 
    'sl', 'sn', 'so', 'sq', 'sr', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr', 'tt', 'uk', 'ur', 'uz', 
    'vi', 'yi', 'yo', 'zh', 'yue'
]
savedir = r'E:\\yodas\\' #external HD
dir = os.getcwd() + "\\"
cache_dir = r"E:/yodas/datasets"

def load_ds(subset):
    while True:
        try:
            ds = load_dataset('espnet/yodas', subset, split="train", trust_remote_code=True, 
                              cache_dir=cache_dir, streaming=streamingvar) # num_proc =
            print('Process subset >>>' , subset)
            making_transcription(ds)
            # pool = Pool(2)
            # pool.map(making_transcription,ds)
        except Exception as e: 
            print(e)
            continue
        else:
            break
        
def load_model():
    model_size = "large-v3"
    # Run on GPU with FP16
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    return model
lm = load_model()

def preprocess_text(text):
    return text.strip().lower().translate(str.maketrans('', '', string.punctuation))

def making_transcription(ds):
    model = lm
     # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    language = subset[0:2] if subset[0:2] in languages else None
    correct_transcriptions = 0
    total = 0
    results_ls_filtered = []
    
    saved_row = Path(dir + subset + '_saved_row.pkl')
    iteration_file = Path(dir + subset + '_saved_index.txt')
    
    start_iteration = 0
    state_dict = None

    # Determine the starting point based on streaming or local mode
    if streamingvar:
        if saved_row.is_file():
            state_dict = open_current_row(subset)
        else:
            state_dict = ds.state_dict()

        print('SAVED CHECKPOINT: ', state_dict)
        ds.load_state_dict(state_dict)
    else:
        if iteration_file.is_file():
            with open(iteration_file, 'rb') as f:
                start_iteration = pickle.load(f)

            # Set ds to start with the last row executed
            rows_tofinish = list(range(start_iteration,ds.num_rows))
            ds = ds.select(rows_tofinish)
            total = start_iteration
        print('SAVED CHECKPOINT: ', total)

    for i in tqdm(ds, desc=f"Processing {subset} Audio Samples"):
        matched_row = {}
        total += 1
        if total < start_iteration:
            continue
        # audio_dict = dict(i['audio']) #somehow i['audio'] gets empty {} after used in pipeline
        audio_array = i['audio']['array']
        
        if streamingvar is True:
            state_dict = ds.state_dict()
            save_current_row(state_dict, subset)

        with torch.no_grad():
            try:
                segments, _ = model.transcribe(audio_array, beam_size=1, language=language)
            except Exception as e: 
                print(e)
        
            expected_text = preprocess_text(i['text'])
            for segment in segments:
                transcribed_text = segment.text

            transcribed_text = preprocess_text(transcribed_text)

            if transcribed_text == expected_text:  # If texts match!
                correct_transcriptions += 1
                results_ls_filtered.append(i['utt_id'])
                
                save_temp(subset, results_ls_filtered)
                audio_compressed = save_compressed_audio(audio_array, i['audio']['sampling_rate'])
                matched_row = {'id': i['id'], 'utt_id': i['utt_id'], 'text': i['text'], 'audio': audio_compressed}

                save_dict(matched_row, subset)
            
            accuracy = correct_transcriptions / total
            torch.cuda.empty_cache()

            # if total == limit_rows:
            #     break
        
        # Save the current iteration number (always for local or stream)
            with open(iteration_file, 'wb') as f:
                pickle.dump(total, f)
      
    # Calculate final accuracy and save results
    final_accuracy = correct_transcriptions / total
    save_results(subset, results_ls_filtered, correct_transcriptions, total, final_accuracy)
    save_ds(subset)  # SAVE DICTIONARY TO DISK, REQUIRES SPACE

def save_ds(subset):
    metadatas = []
    # Loading all objects from the file
    try:
        file = open(subset + '_data.pkl', 'rb')
    except FileNotFoundError:
        print('_data file not found')
    else:
        with file:
            metadatas.append(pickle.load(file))
            loaded_data = Dataset.from_list(metadatas)
            loaded_data.save_to_disk(savedir + subset)
            print('Dataset saved successfully')

def save_results(subset, results_ls_filtered, ct, tot, acc):
    # Save IDs to csv file to read later
    # import csv
    with open(savedir + subset + '_'+ str(ct) + '_'+ str(tot) + '_'+ str(round(acc,2)) + '.txt', 'w') as f:
        # Join the list elements into a single string with a newline character
        data_to_write = '\n'.join(results_ls_filtered)
        
        # Write the data to the file
        f.write(data_to_write)
        
        print("Results file written successfully")

def save_temp(subset, results_ls_filtered):
    # Save IDs to csv file to read later
    with open(dir + subset + '.txt', 'w') as f:
        data_to_write = '\n'.join(results_ls_filtered)
        
        # Write the data to the file
        f.write(data_to_write)
        
        print("Row written successfully")

def save_compressed_audio(audio_data, sample_rate):
    # Save audio data in compressed format (e.g., WAV or FLAC) in memory
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
    return buffer.getvalue()

def save_current_row(state_dict, subset):
    with open(subset + '_saved_row.pkl', 'wb') as f:
        pickle.dump(state_dict, f)

def open_current_row(subset):
    with open(subset + '_saved_row.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
        return loaded_dict

    #Original dataset in streaming is filter only for the rows that matched
    # THIS LINE CAN BE USED AFTER ds = load_dataset('espnet/yodas'...
    # filtered_dataset = ds.filter(lambda x: x["id"] in results_ls_filtered)

def save_dict(matched_row, subset):
    # Append the dictionary to the file
    with open(subset + '_data.pkl', 'ab') as file:
        pickle.dump(matched_row, file)

def process_subset(subs):
    configs = [subs]

# BY SIZE            
#                ,'es102','es104','es105','es103','en108','en126','en107','en119','en106','es101','en122','en121','en120','en103',
# 'en104','es100','en116','en101','en112','en113','ru104','ru102','en111','en118','en102','en117','en124','en109','en114','en115',
# 'en125','en110','ru103','ru100','ru101','en105','en123','fr102','ru105','fr100','fr101','en004','ko101','pt100','pt102','en002',
# 'pt101','en001','ko100','ko102','en003','en000','de100','de101','it100','ru000','vi100','id100','it101','vi101','tr100','en005',
# 'es000','id101','pt103','ru106','fr103','de000','ja100','ko000','es106','ko103','nl100','fr000','en127','pt000','it000','id000',
# 'ja000','de102','vi000','pl000','ru001','tr000','th000','hi000','uk100','nl000','uk000','zh000','ar000','fi000','hu000','cs000',
# 'iw000','no000','sv000','el000','ro000','ca000','ta000','be000','bg000','fa000','sk000','ms000','da000','bn000','ka000','hr000',
# 'sl000','ur000','eu000','lt000','sr000','et000','ky000','ml000','eo000','gl000','bs000','la000','mr000','te000','mk000','uz000',
# 'cy000','is000','si000','km000','az000','kk000','sq000','so000','hi100','lv000','kn000','my000','ne000','mn000','gu000','ku000',
# 'sw000','th100','hy000','pa000','ga000','mi000','jv000','ht000','ps000','am000','af000','qu000','bo000','br000','rw000','as000',
# 'or000','ab000','sa000','ti000','yo000','tg000','sh000','ak000','lo000','vo000','rm000','ln000','fo000','gn000','aa000','mg000',
# 'oc000','om000','zu000','ie000','xh000','tn000','lb000','ha000','sm000','ug000','ig000','ia000','yi000','wo000','sd000','tk000',
# 'fy000','dz000','iu000','ho000','tt000','co000','ee000','su000','na000','ff000','ay000','ba000','gd000','fj000','ks000','sn000',
# 'bh000','bi000','sc000','cr000','kl000','ik000','rn000','lg000','ve000','st000','nv000','bm000','nd000','ts000','ki000','to000',
# 'sg000']
    
    Parallel(n_jobs=1, backend="multiprocessing")(delayed(load_ds)(subset) for subset in configs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Subset, streaming T/F"
    )
    parser.add_argument("--sub", default='en100', type=str)
    parser.add_argument('--stream', dest='stream', action='store_true',
                    help='Set the stream value to True.')
    parser.add_argument('--no-stream', dest='stream', action='store_false',
                    help='Set the stream value to False.')
    parser.set_defaults(stream=True)
    args = parser.parse_args()

    subset = args.sub
    streamingvar = args.stream
    
    process_subset(subset)

    # Parallel(n_jobs=1, backend="multiprocessing")(delayed(load_ds)(subset) for subset in configs)

    # pool_obj = mp.Pool(2)
    # pool_obj.map(load_ds, configs)

    # with get_context("spawn").Pool(processes=2) as pool:
    #     pool.map(load_ds, configs)

    # for config in configs:
    #     load_ds(config)