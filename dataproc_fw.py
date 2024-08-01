from tabnanny import verbose
from networkx import johnson, node_disjoint_paths
from datasets import load_dataset, Dataset
import sys
import string
from tqdm import tqdm  # For progress bar
import logging
from difflib import SequenceMatcher
import io
import string
import soundfile as sf
from difflib import SequenceMatcher
from faster_whisper import WhisperModel
from joblib import Parallel, delayed
# from utils import get_total_gpu_memory
MODEL_MEMORY = 5000
from multiprocessing import cpu_count
from multiprocessing import get_context
from logging import getLogger
LOG = getLogger(__name__)
LOG.setLevel(logging.WARNING)
import torch
import torch.multiprocessing as mp
import gc


languages = [
    'af', 'am', 'ar', 'as', 'az', 'ba', 'be', 'bg', 'bn', 'bo', 'br', 'bs', 'ca', 'cs', 'cy', 'da', 'de', 'el', 'en', 
    'es', 'et', 'eu', 'fa', 'fi', 'fo', 'fr', 'gl', 'gu', 'ha', 'haw', 'he', 'hi', 'hr', 'ht', 'hu', 'hy', 'id', 'is', 
    'it', 'ja', 'jw', 'ka', 'kk', 'km', 'kn', 'ko', 'la', 'lb', 'ln', 'lo', 'lt', 'lv', 'mg', 'mi', 'mk', 'ml', 'mn', 
    'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'nn', 'no', 'oc', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'sa', 'sd', 'si', 'sk', 
    'sl', 'sn', 'so', 'sq', 'sr', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr', 'tt', 'uk', 'ur', 'uz', 
    'vi', 'yi', 'yo', 'zh', 'yue'
]

savedir = r'C:\\Users\\hudso\\Downloads\\HudsonAI\\subsets\\'


def load_ds(subset):
    # #For Google Collab
    # import os, sys
    # from google.colab import drive
    # drive.mount('/content/drive')
    # nb_path = '/content/notebooks'
    # os.symlink('/content/drive/My Drive/Colab Notebooks', nb_path)
    # sys.path.insert(0, nb_path)

    # !pip install --target=$nb_path git+https://github.com/huggingface/transformers.git accelerate datasets[audio]

    # !pip install --target=$nb_path datasets

    # pip install git+https://github.com/huggingface/transformers.git accelerate datasets[audio]

    # pip install datasets

    # FOR EXECUTING ALL THE FILES AND GENERATING MANY CSVs with filtered rows
    '''

    LOAD MODEL (function)
    First execute a for loop with all the shards/subsets (function)
    For each one, LOAD DATA and execute the pipeline
    Generate csv
    decide 


    '''

    # all folders in espnet/yodas (224 subsets)
    # done 'aa000', 'ab000', 'af000', 'ak000', 'am000', 'ar000', 'as000', 'ay000', 'az000', 'ba000', 'be000', 'bg000'
    # 'bh000', 'bi000', 'bm000', 'bn000', 'bo000', 'br000', bs000',
    
    # configs = ['aa000', 'ab000', 'af000', 'ak000', 'am000', 'ar000', 'as000', 'ay000', 'az000', 'ba000', 'be000', 'bg000',
    # 'bh000', 'bi000', 'bm000', 'bn000', 'bo000', 'br000', 'bs000', 'ca000', 'co000', 'cr000', 'cs000', 'cy000', 
    # 'da000', 'de000', 'de100', 'de101', 'de102', 'dz000', 'ee000', 'el000', 'en000', 'en001', 'en002', 'en003', 'en004', 
    # 'en005', 'en100', 'en101', 'en102', 'en103', 'en104', 'en105', 'en106', 'en107', 'en108', 'en109', 'en110', 'en111', 
    # 'en112', 'en113', 'en114', 'en115', 'en116', 'en117', 'en118', 'en119', 'en120', 'en121', 'en122', 'en123', 'en124', 
    # 'en125', 'en126', 'en127', 'eo000', 'es000', 'es100', 'es101', 'es102', 'es103', 'es104', 'es105', 'es106', 'et000', 
    # 'eu000', 'fa000', 'ff000', 'fi000', 'fj000', 'fo000', 'fr000', 'fr100', 'fr101', 'fr102', 'fr103', 'fy000', 'ga000', 
    # 'gd000', 'gl000', 'gn000', 'gu000', 'ha000', 'hi000', 'hi100', 'ho000', 'hr000', 'ht000', 'hu000', 'hy000', 'ia000', 
    # 'id000', 'id100', 'id101', 'ie000', 'ig000', 'ik000', 'is000', 'it000', 'it100', 'it101', 'iu000', 'iw000', 'ja000', 
    # 'ja100', 'jv000', 'ka000', 'ki000', 'kk000', 'kl000', 'km000', 'kn000', 'ko000', 'ko100', 'ko101', 'ko102', 'ko103', 
    # 'ks000', 'ku000', 'ky000', 'la000', 'lb000', 'lg000', 'ln000', 'lo000', 'lt000', 'lv000', 'mg000', 'mi000', 'mk000', 
    # 'ml000', 'mn000', 'mr000', 'ms000', 'my000', 'na000', 'nd000', 'ne000', 'nl000', 'nl100', 'no000', 'nv000', 'oc000', 
    # 'om000', 'or000', 'pa000', 'pl000', 'ps000', 'pt000', 'pt100', 'pt101', 'pt102', 'pt103', 'qu000', 'rm000', 'rn000', 
    # 'ro000', 'ru000', 'ru001', 'ru100', 'ru101', 'ru102', 'ru103', 'ru104', 'ru105', 'ru106', 'rw000', 'sa000', 'sc000', 
    # 'sd000', 'sg000', 'sh000', 'si000', 'sk000', 'sl000', 'sm000', 'sn000', 'so000', 'sq000', 'sr000', 'st000', 'su000', 
    # 'sv000', 'sw000', 'ta000', 'te000', 'tg000', 'th000', 'th100', 'ti000', 'tk000', 'tn000', 'to000', 'tr000', 'tr100', 
    # 'ts000', 'tt000', 'ug000', 'uk000', 'uk100', 'ur000', 'uz000', 've000', 'vi000', 'vi100', 'vi101', 'vo000', 'wo000', 
    # 'xh000', 'yi000', 'yo000', 'zh000', 'zu000']
    # configs = ['xh000']

    # for subset in configs:
    while True:
        try:
            ds = load_dataset('espnet/yodas', subset, split="train", streaming=True, trust_remote_code=True, )
            print('Read this >>>' , subset)
            print(ds)
            making_transcription(subset, ds, lm)
            # this STREAMING loading will finish quickly, can load only one by one
            #subset = 'mk000'
        except Exception as e: 
            print(e)
            continue
        else:
            break


# def load_ds(lm, subset):
#     while True:
#         try:
#             ds = load_dataset('espnet/yodas', subset, split="train", streaming=True, trust_remote_code=True)
#             print(ds)
#             print('Read this:', subset)
#             making_transcription(subset, ds, lm)
#         except Exception as e:
#             print(e)
#             continue
#         else:
#             print('haremos break')
#             break
        
def load_model():
    model_size = "large-v3"

    # Run on GPU with FP16
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")

    return model
lm = load_model()


# change array to audiofile-like
# def array_to_filelike(audio_array, sampling_rate):
#     filelike = io.BytesIO()
#     sf.write(filelike, audio_array, sampling_rate, format='WAV')
#     filelike.seek(0)
#     return filelike

def preprocess_text(text):
    return text.strip().lower().translate(str.maketrans('', '', string.punctuation))

def making_transcription(subset, ds, model):
    language = subset[0:2] if subset[0:2] in languages else None

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # def evaluate_transcriptions(sample, pipe):
    # if len(results_dict) == 0: 
    correct_transcriptions = 0
    total = 0
    transcribed_text = ''     #needed if no segments come
    # results_ls = []
    # results_dict = []
    results_ls_filtered = []
    metadatas = []

    for i in tqdm(ds, desc=f"Processing {subset} Audio Samples"):
        # audio_dict = dict(i['audio']) #somehow i['audio'] gets empty {} after used in pipeline
        audio_array = i['audio']['array']
        # audio_array = audio_info['array']
        # sampling_rate = audio_info['sampling_rate']

        # audio_filelike = array_to_filelike(audio_array, sampling_rate)
        with torch.no_grad():
            try:
                segments, _ = model.transcribe(audio_array, beam_size=1, language=language)
            except Exception as e: print(e)
                # print(f"Problems with row: {i['id']}")
                # sys.exit(1)

            total += 1
            expected_text = preprocess_text(i['text'])
            for segment in segments:
                transcribed_text = segment.text

            transcribed_text = preprocess_text(transcribed_text)

            # ratio = SequenceMatcher(None, expected_text, transcribed_text).ratio()

            if transcribed_text == expected_text: # if texts match!
                # matched = 1
                correct_transcriptions += 1
                
                # if i['id'] not in results_ls_filtered:
                results_ls_filtered.append(i['id'])
                save_temp(subset, results_ls_filtered)
                # else: 
                #     break
                # Originally dataset keeps reading after it ends, for big subsets, 
                # with this rule, when it finds the same id (repeating), will break and go to the next
                # metadatas.append({'id': i['id'], 'utt_id': i['utt_id'], 
                #                 'expected': i['text'], 'transcribed': transcribed_text, 'audio': audio_dict})
            # else:
            #     matched = 0
            # print(i['id'])
            # if i['id'] not in results_ls_filtered:
            #     print('guarda file?')
            #     save_temp(subset, results_ls_filtered)
            # else:
            #     print('ya estaba y rompe, pasa a siguiente', subset)
            #     break
            
            accuracy = correct_transcriptions / total
            # results_ls.append([i['id'], i['text'], transcribed_text, accuracy, ratio])
            # results_dict.append({'id': i['id'], 'exptext': i['text'], 'tratext': transcribed_text, 'acc': accuracy, 'ratio': ratio})

            # logging.info(f"Processed {total} samples. Current accuracy: {accuracy:.2%}")
            torch.cuda.empty_cache()
            # gc.collect()
            # break #just test all subsets
    final_accuracy = correct_transcriptions / total
    # logging.info(f"Final Accuracy: {final_accuracy:.2%}")
    # metadatas_ds = Dataset.from_list(metadatas)

    # If needed, print the final accuracy
    # print(f"Final Accuracy: {final_accuracy:.2%}")

    # for i in results_ls:
    #     print(i)
    #     print(len(results_ls))
    #     break

    # for i in metadata:
    #     print(i)
    #     print(len(metadata))
    #     break

    save_csv(subset, results_ls_filtered, correct_transcriptions, total, final_accuracy)
    # save_ds(subset, metadatas_ds) # SAVE DICTIONARY TO DISK, REQUIRES SPACE

def save_ds(subset, metadatas_ds):
    metadatas_ds.save_to_disk(subset+'.hf')

def save_csv(subset, results_ls_filtered, ct, tot, acc):
    # Save IDs to csv file to read later
    # import csv

    with open(savedir + subset + '_'+ str(ct) + '_'+ str(tot) + '_'+ str(round(acc,2)) + '.txt', 'w') as f:

        # using csv.writer method from CSV package
        # write = csv.writer(f)

        # write.writerow(results_ls_filtered)
     
        # write elements of list
        # Join the list elements into a single string with a newline character
        data_to_write = '\n'.join(results_ls_filtered)
        
        # Write the data to the file
        f.write(data_to_write)
        
        print("File written successfully")

def save_temp(subset, results_ls_filtered):
    # Save IDs to csv file to read later
    # import csv

    with open(savedir + subset + '.txt', 'w') as f:

        # using csv.writer method from CSV package
        # write = csv.writer(f)

        # write.writerow(results_ls_filtered)
     
        # write elements of list
        # Join the list elements into a single string with a newline character
        # print('entra a save temp')
        data_to_write = '\n'.join(results_ls_filtered)
        
        # Write the data to the file
        f.write(data_to_write)
        
        print("Row written successfully")

def open_csv():
    # OPEN csv file to start reading from here and stream the dataset with these IDs only
    import csv

    with open(subset + 'FW.csv', newline='') as f:
        reader = csv.reader(f)
        results_ls_filtered = list(reader)[0]

    print(len(results_ls_filtered))


    #Original dataset in streaming is filter only for the rows that matched

    # THIS LINE CAN BE USED AFTER ds = load_dataset('espnet/yodas'...
    # filtered_dataset = ds.filter(lambda x: x["id"] in results_ls_filtered)

    # print(filtered_dataset)
    # print(next(iter(filtered_dataset)))

    # for i in results_ls_filtered:
    #     print(i)
    #     break

if __name__ == '__main__':
    # load_ds(load_model())

# if __name__ == '__main__':
#     configs = ['oc000', 'es000']  # Add your full list of subsets here
#     lm = load_model()

#     with Pool(processes=len(configs)) as pool:
#         pool.starmap(load_ds, [(lm, subset) for subset in configs])

    # configs = ['ar000', 'be000', 'bg000',
    #     'bh000', 'bi000', 'bm000', 'bn000', 'bo000', 'br000', 'bs000', 'ca000', 'co000', 'cr000', 'cs000', 'cy000', 
    #     'da000', 'de000', 'de100', 'de101', 'de102', 'dz000', 'ee000', 'el000', 'en000', 'en001', 'en002', 'en003', 'en004', 
    #     'en005', 'en100', 'en101', 'en102', 'en103', 'en104', 'en105', 'en106', 'en107', 'en108', 'en109', 'en110', 'en111', 
    #     'en112', 'en113', 'en114', 'en115', 'en116', 'en117', 'en118', 'en119', 'en120', 'en121', 'en122', 'en123', 'en124', 
    #     'en125', 'en126', 'en127', 'eo000', 'es000', 'es100', 'es101', 'es102', 'es103', 'es104', 'es105', 'es106', 'et000', 
    #     'eu000', 'fa000',  'fi000',  'fr000', 'fr100', 'fr101', 'fr102', 'fr103',  'gl000',  'hi000',  'hr000',  'hu000',  
    #     'id000', 'id100', 'id101',  'is000', 'it000', 'it100', 'it101', 'iw000', 'ja000', 
    #     'ja100',  'ka000', 'ki000',  'kl000', 'km000', 'ko000', 'ko100', 'ko101', 'ko102', 'ko103', 
    #      'ky000', 'la000', 'lt000', 'lv000',  'mk000', 
    #     'ml000',  'mr000', 'ms000', 'my000',  'nd000', 'ne000', 'nl000', 'nl100', 'no000','oc000', 
    #     'pl000',  'pt000', 'pt100', 'pt101', 'pt102', 'pt103',  
    #     'ro000', 'ru000','ru100', 'ru101', 'ru102', 'ru103', 'ru104', 'ru105', 'ru106',  'sg000', 'sh000', 'si000', 'sk000', 'sl000', 'sm000',  'sq000', 'sr000', 
    #     'sv000', 'sw000', 'ta000', 'te000', 'th000', 'th100','to000', 'tr000', 'tr100', 
    #     'ts000', 'uk000', 'uk100', 'ur000',  'vi000', 'vi100',  
    #     'xh000', 'zh000']
    # configs = ['oc000','xh000', 'bo000','sm000','sh000','kl000','bo000','bi000','bh000']  # Add your full list of subsets here
    # configs = ['is000', 'bs000']
    # configs = ['km000'] # SLOW
    # configs = [ 
    #     'de101', 'de102', 'dz000', 'ee000', 'el000', 'en000', 'en001', 'en002', 'en003', 'en004', 
    #     'en005', 'en100', 'en101', 'en102', 'en103', 'en104', 'en105', 'en106', 'en107', 'en108', 'en109', 'en110', 'en111', 
    #     'en112', 'en113', 'en114', 'en115', 'en116', 'en117', 'en118', 'en119', 'en120', 'en121', 'en122', 'en123', 'en124', 
    #     'en125', 'en126', 'en127', 'eo000', 'es000', 'es100', 'es101', 'es102', 'es103', 'es104', 'es105', 'es106', 'et000', 
    #     'eu000']

    configs = ["as000",
"rw000",
"br000",
"bo000",
"qu000",
"af000",
"am000",
"ps000"]

    # configs = ['es102', 'es103', 'es104', 'es105', 'es106'] 'da000', 'de000', 'de100', 
    # 'en110' --- 140K??? 
    # 'en111',
    #  'ff000', 'fj000','fo000','fy000', 'ga000', 
    #     'gd000','gn000', 'gu000', 'ru001'

    

    #     # Execute in parallel
    # memory = torch.cuda.mem_get_info(device='cuda:0')[0]//1024 ** 2
    # n_jobs = min(
    #     max(
    #         memory
    #         // MODEL_MEMORY
    #         if memory is not None
    #         else 1,
    #         1,
    #     ),
    #     cpu_count(),
    # ) 
    # # LOG.info(f"n_jobs automatically set to {n_jobs}, memory: {memory} MiB")
    
    # n_jobs = min(len(configs) // 16 + 1, n_jobs)
    # # print(n_jobs)
    # Parallel(n_jobs=2, backend="multiprocessing")(delayed(load_ds)(subset) for subset in configs)

    # pool_obj = mp.Pool(2)
    # pool_obj.map(load_ds, configs)

    # with get_context("spawn").Pool(processes=1) as pool:
    #     pool.map(load_ds, configs)

    for config in configs:
        load_ds(config)