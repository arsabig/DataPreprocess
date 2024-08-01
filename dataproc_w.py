# import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sys
import string
from tqdm import tqdm  # For progress bar
import logging
from difflib import SequenceMatcher
from multiprocessing import get_context
import torch
import gc
import torch.multiprocessing as mp

languages = [
    'en', 'zh', 'de', 'es', 'ru', 'ko', 'fr', 'ja', 'pt', 'tr', 'pl', 'ca', 'nl', 'ar', 
    'sv', 'it', 'id', 'hi', 'fi', 'vi', 'he', 'uk', 'el', 'ms', 'cs', 'ro', 'da', 'hu', 
    'ta', 'no', 'th', 'ur', 'hr', 'bg', 'lt', 'la', 'mi', 'ml', 'cy', 'sk', 'te', 'fa', 
    'lv', 'bn', 'sr', 'az', 'sl', 'kn', 'et', 'mk', 'br', 'eu', 'is', 'hy', 'ne', 'mn', 
    'bs', 'kk', 'sq', 'sw', 'gl', 'mr', 'pa', 'si', 'km', 'sn', 'yo', 'so', 'af', 'oc', 
    'ka', 'be', 'tg', 'sd', 'gu', 'am', 'yi', 'lo', 'uz', 'fo', 'ht', 'ps', 'tk', 'nn', 
    'mt', 'sa', 'lb', 'my', 'bo', 'tl', 'mg', 'as', 'tt', 'haw', 'ln', 'ha', 'ba', 'jw', 
    'su'
]


def load_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,                                                      
        chunk_length_s=30,
        batch_size=1,
        return_timestamps=False,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe
model = load_model()

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

    #all folders in espnet/yodas (224 subsets)
    # configs = ['aa000', 'ab000', 'af000', 'ak000', 'am000', 'ar000', 'as000', 'ay000', 'az000', 'ba000', 'be000', 'bg000', 'bh000', 'bi000', 'bm000', 'bn000', 'bo000', 'br000', 'bs000', 'ca000', 'co000', 'cr000', 'cs000', 'cy000', 'da000', 'de000', 'de100', 'de101', 'de102', 'dz000', 'ee000', 'el000', 'en000', 'en001', 'en002', 'en003', 'en004', 'en005', 'en100', 'en101', 'en102', 'en103', 'en104', 'en105', 'en106', 'en107', 'en108', 'en109', 'en110', 'en111', 'en112', 'en113', 'en114', 'en115', 'en116', 'en117', 'en118', 'en119', 'en120', 'en121', 'en122', 'en123', 'en124', 'en125', 'en126', 'en127', 'eo000', 'es000', 'es100', 'es101', 'es102', 'es103', 'es104', 'es105', 'es106', 'et000', 'eu000', 'fa000', 'ff000', 'fi000', 'fj000', 'fo000', 'fr000', 'fr100', 'fr101', 'fr102', 'fr103', 'fy000', 'ga000', 'gd000', 'gl000', 'gn000', 'gu000', 'ha000', 'hi000', 'hi100', 'ho000', 'hr000', 'ht000', 'hu000', 'hy000', 'ia000', 'id000', 'id100', 'id101', 'ie000', 'ig000', 'ik000', 'is000', 'it000', 'it100', 'it101', 'iu000', 'iw000', 'ja000', 'ja100', 'jv000', 'ka000', 'ki000', 'kk000', 'kl000', 'km000', 'kn000', 'ko000', 'ko100', 'ko101', 'ko102', 'ko103', 'ks000', 'ku000', 'ky000', 'la000', 'lb000', 'lg000', 'ln000', 'lo000', 'lt000', 'lv000', 'mg000', 'mi000', 'mk000', 'ml000', 'mn000', 'mr000', 'ms000', 'my000', 'na000', 'nd000', 'ne000', 'nl000', 'nl100', 'no000', 'nv000', 'oc000', 'om000', 'or000', 'pa000', 'pl000', 'ps000', 'pt000', 'pt100', 'pt101', 'pt102', 'pt103', 'qu000', 'rm000', 'rn000', 'ro000', 'ru000', 'ru001', 'ru100', 'ru101', 'ru102', 'ru103', 'ru104', 'ru105', 'ru106', 'rw000', 'sa000', 'sc000', 'sd000', 'sg000', 'sh000', 'si000', 'sk000', 'sl000', 'sm000', 'sn000', 'so000', 'sq000', 'sr000', 'st000', 'su000', 'sv000', 'sw000', 'ta000', 'te000', 'tg000', 'th000', 'th100', 'ti000', 'tk000', 'tn000', 'to000', 'tr000', 'tr100', 'ts000', 'tt000', 'ug000', 'uk000', 'uk100', 'ur000', 'uz000', 've000', 'vi000', 'vi100', 'vi101', 'vo000', 'wo000', 'xh000', 'yi000', 'yo000', 'zh000', 'zu000']
    # configs = ['xh000']

    # for i in configs:
    #     subset = i
    #     ds = load_dataset('espnet/yodas', subset, split="train", streaming=True, trust_remote_code=True)
    #     # this STREAMING loading will finish quickly, can load only one by one
    #     #subset = 'mk000'
    #     print('>>> ', i)
    #     making_transcription(i, ds, lm)
    while True:
        try:
            ds = load_dataset('espnet/yodas', subset, split="train", streaming=True, trust_remote_code=True)
            print(ds)
            print('Read this:', subset)
            making_transcription(subset, ds, model)
        except Exception as e:
            print(e)
            continue
        else:
            break

def preprocess_text(text):
    return text.strip().lower().translate(str.maketrans('', '', string.punctuation))

def making_transcription(subset, ds, pipe):
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # def evaluate_transcriptions(sample, pipe):
    # if len(results_dict) == 0: 
    language = subset[0:2] if subset[0:2] in languages else None
    correct_transcriptions = 0
    total = 0
    results_ls = []
    # results_dict = []
    results_ls_filtered = []
    metadatas = []

    for i in tqdm(ds, desc=f"Processing {subset} Audio Samples"):
        audio_dict = dict(i['audio'])        
        with torch.no_grad():
            try:
                result = pipe(i['audio'], generate_kwargs={"language": language})
            except Exception as e: print(e)
                # print(f"Problems with row: {i['id']}")
                # sys.exit(1)

            total += 1
            expected_text = preprocess_text(i['text'])
            transcribed_text = preprocess_text(result['text'])

            ratio = SequenceMatcher(None, expected_text, transcribed_text).ratio()

            if transcribed_text == expected_text: # if texts match!
                matched = 1
                correct_transcriptions += 1
                results_ls_filtered.append(i['id'])
                # metadatas.append({'id': i['id'], 'utt_id': i['utt_id'], 
                                    # 'expected': i['text'], 'transcribed': transcribed_text, 'audio': audio_dict})
            else:
                matched = 0

            accuracy = correct_transcriptions / total
            results_ls.append([i['id'], i['text'], result['text'], matched, ratio])
            # results_dict.append({'id': i['id'], 'exptext': i['text'], 'tratext': result['text'], 'acc': matched, 'ratio': ratio})

            logging.info(f"Processed {total} samples. Current accuracy: {accuracy:.2%}")
        torch.cuda.empty_cache()
        gc.collect()

    final_accuracy = correct_transcriptions / total
    logging.info(f"Final Accuracy: {final_accuracy:.2%}")
    # metadatas_ds = Dataset.from_list(metadatas)

    # If needed, print the final accuracy
    # print(f"Final Accuracy: {final_accuracy:.2%}")

    # for i in results_ls:
    #     print(i)
    #     print(len(results_ls))
    #     break

    save_csv(subset, results_ls_filtered)
    # save_ds(subset, metadatas_ds) # SAVE TO DISK, REQUIRES SPACE

def save_ds(subset, metadatas_ds):
    metadatas_ds.save_to_disk(subset+'.hf')

def save_csv(subset, results_ls_filtered):
    # Save IDs to csv file to read later
    # import csv

    with open(subset + 'W.csv', 'w', newline ='') as f:

        # using csv.writer method from CSV package
        # write = csv.writer(f)

        # write.writerow(results_ls_filtered)
        data_to_write = '\n'.join(results_ls_filtered)
        
        # Write the data to the file
        f.write(data_to_write)
        
        print("File written successfully")

def open_csv():
    # OPEN csv file to start reading from here and stream the dataset with these IDs only
    import csv

    with open(subset + '.csv', newline='') as f:
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
    # if load_model():
    #     load_ds(load_model())

    # configs = ['xh000']
    # configs = ['ar000', 'be000', 'bg000',
    #     'bh000', 'bi000', 'bm000', 'bn000', 'bo000', 'br000', 'bs000', 'ca000', 'co000', 'cr000', 'cs000', 'cy000', 
    #     'da000', 'de000', 'de100', 'de101', 'de102', 'dz000', 'ee000', 'el000', 'en000', 'en001', 'en002', 'en003', 'en004', 
    #     'en005', 'en100', 'en101', 'en102', 'en103', 'en104', 'en105', 'en106', 'en107', 'en108', 'en109', 'en110', 'en111', 
    #     'en112', 'en113', 'en114', 'en115', 'en116', 'en117', 'en118', 'en119', 'en120', 'en121', 'en122', 'en123', 'en124', 
    #     'en125', 'en126', 'en127', 'eo000', 'es000', 'es100', 'es101', 'es102', 'es103', 'es104', 'es105', 'es106', 'et000', 
    #     'eu000', 'fa000', 'ff000', 'fi000', 'fj000', 'fo000', 'fr000', 'fr100', 'fr101', 'fr102', 'fr103', 'fy000', 'ga000', 
    #     'gd000', 'gl000', 'gn000', 'gu000', 'ha000', 'hi000', 'hi100', 'ho000', 'hr000', 'ht000', 'hu000', 'hy000', 'ia000', 
    #     'id000', 'id100', 'id101', 'ie000', 'ig000', 'ik000', 'is000', 'it000', 'it100', 'it101', 'iu000', 'iw000', 'ja000', 
    #     'ja100', 'jv000', 'ka000', 'ki000', 'kk000', 'kl000', 'km000', 'kn000', 'ko000', 'ko100', 'ko101', 'ko102', 'ko103', 
    #     'ks000', 'ku000', 'ky000', 'la000', 'lb000', 'lg000', 'ln000', 'lo000', 'lt000', 'lv000', 'mg000', 'mi000', 'mk000', 
    #     'ml000', 'mn000', 'mr000', 'ms000', 'my000', 'na000', 'nd000', 'ne000', 'nl000', 'nl100', 'no000', 'nv000', 'oc000', 
    #     'om000', 'or000', 'pa000', 'pl000', 'ps000', 'pt000', 'pt100', 'pt101', 'pt102', 'pt103', 'qu000', 'rm000', 'rn000', 
    #     'ro000', 'ru000', 'ru001', 'ru100', 'ru101', 'ru102', 'ru103', 'ru104', 'ru105', 'ru106', 'rw000', 'sa000', 'sc000', 
    #     'sd000', 'sg000', 'sh000', 'si000', 'sk000', 'sl000', 'sm000', 'sn000', 'so000', 'sq000', 'sr000', 'st000', 'su000', 
    #     'sv000', 'sw000', 'ta000', 'te000', 'tg000', 'th000', 'th100', 'ti000', 'tk000', 'tn000', 'to000', 'tr000', 'tr100', 
    #     'ts000', 'tt000', 'ug000', 'uk000', 'uk100', 'ur000', 'uz000', 've000', 'vi000', 'vi100', 'vi101', 'vo000', 'wo000', 
    #     'xh000', 'yi000', 'yo000', 'zh000', 'zu000']

    configs = ['h100']
    # with get_context("spawn").Pool(processes=3) as pool:
    #     pool.map(load_ds, configs)
    
    
    pool_obj = mp.Pool(2)
    pool_obj.map(load_ds, configs)