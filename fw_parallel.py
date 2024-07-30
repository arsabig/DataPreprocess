# import string
# import logging
# from difflib import SequenceMatcher
# from tqdm import tqdm
# from datasets import load_dataset, Dataset
# from faster_whisper import WhisperModel
# from joblib import Parallel, delayed
# import torch, gc

# languages = [
#     'af', 'am', 'ar', 'as', 'az', 'ba', 'be', 'bg', 'bn', 'bo', 'br', 'bs', 'ca', 'cs', 'cy', 'da', 'de', 'el', 'en', 
#     'es', 'et', 'eu', 'fa', 'fi', 'fo', 'fr', 'gl', 'gu', 'ha', 'haw', 'he', 'hi', 'hr', 'ht', 'hu', 'hy', 'id', 'is', 
#     'it', 'ja', 'jw', 'ka', 'kk', 'km', 'kn', 'ko', 'la', 'lb', 'ln', 'lo', 'lt', 'lv', 'mg', 'mi', 'mk', 'ml', 'mn', 
#     'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'nn', 'no', 'oc', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'sa', 'sd', 'si', 'sk', 
#     'sl', 'sn', 'so', 'sq', 'sr', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr', 'tt', 'uk', 'ur', 'uz', 
#     'vi', 'yi', 'yo', 'zh', 'yue'
# ]

# def load_ds_and_make_transcription(subset):
#     try:
#         ds = load_dataset('espnet/yodas', subset, split="train", streaming=True, trust_remote_code=True)
#         print(f'Read dataset: {subset}')
#         model = WhisperModel("large-v3", device="cuda", compute_type="float16")
#         making_transcription(subset, ds, model)
#     except Exception as e:
#         print(f'Error loading dataset {subset}: {e}')

# def preprocess_text(text):
#     return text.strip().lower().translate(str.maketrans('', '', string.punctuation))

# def making_transcription(subset, ds, model):
#     language = subset[0:2] if subset[0:2] in languages else None

#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#     correct_transcriptions = 0
#     total = 0
#     results_ls_filtered = []
#     metadatas = []

#     for i in tqdm(ds, desc=f"Processing {subset} Audio Samples"):
#         audio_dict = dict(i['audio'])
#         audio_array = i['audio']['array']
#         with torch.no_grad():
#             try:
#                 segments, _ = model.transcribe(audio_array, beam_size=1, language=language)
#             except Exception as e: 
#                 print(f"Error transcribing audio in {subset}: {e}")
#                 continue

#             total += 1
#             expected_text = preprocess_text(i['text'])
#             transcribed_text = ' '.join([preprocess_text(segment.text) for segment in segments])

#             if transcribed_text == expected_text:
#                 correct_transcriptions += 1
#                 results_ls_filtered.append(i['id'])
#                 metadatas.append({
#                     'id': i['id'], 'utt_id': i['utt_id'], 
#                     'expected': i['text'], 'transcribed': transcribed_text, 'audio': audio_dict
#                 })

#             accuracy = correct_transcriptions / total
#             logging.info(f"Processed {total} samples. Current accuracy: {accuracy:.2%}")
#         torch.cuda.empty_cache()
#         gc.collect()

#     final_accuracy = correct_transcriptions / total
#     print(f"Final Accuracy for {subset}: {final_accuracy:.2%}")

#     metadatas_ds = Dataset.from_list(metadatas)
#     save_csv(subset, results_ls_filtered)

# def save_csv(subset, results_ls_filtered):
#     import csv
#     with open(subset + '.csv', 'w', newline='') as f:
#         write = csv.writer(f)
#         write.writerow(results_ls_filtered)

# if __name__ == '__main__':
#     # configs = ['aa000', 'ab000', 'af000', 'ak000', 'am000', 'ar000', 'as000', 'ay000', 'az000', 'ba000', 'be000', 'bg000',
#     # 'bh000', 'bi000', 'bm000', 'bn000', 'bo000', 'br000', 'bs000', 'ca000', 'co000', 'cr000', 'cs000', 'cy000', 
#     # 'da000', 'de000', 'de100', 'de101', 'de102', 'dz000', 'ee000', 'el000', 'en000', 'en001', 'en002', 'en003', 'en004', 
#     # 'en005', 'en100', 'en101', 'en102', 'en103', 'en104', 'en105', 'en106', 'en107', 'en108', 'en109', 'en110', 'en111', 
#     # 'en112', 'en113', 'en114', 'en115', 'en116', 'en117', 'en118', 'en119', 'en120', 'en121', 'en122', 'en123', 'en124', 
#     # 'en125', 'en126', 'en127', 'eo000', 'es000', 'es100', 'es101', 'es102', 'es103', 'es104', 'es105', 'es106', 'et000', 
#     # 'eu000', 'fa000', 'ff000', 'fi000', 'fj000', 'fo000', 'fr000', 'fr100', 'fr101', 'fr102', 'fr103', 'fy000', 'ga000', 
#     # 'gd000', 'gl000', 'gn000', 'gu000', 'ha000', 'hi000', 'hi100', 'ho000', 'hr000', 'ht000', 'hu000', 'hy000', 'ia000', 
#     # 'id000', 'id100', 'id101', 'ie000', 'ig000', 'ik000', 'is000', 'it000', 'it100', 'it101', 'iu000', 'iw000', 'ja000', 
#     # 'ja100', 'jv000', 'ka000', 'ki000', 'kk000', 'kl000', 'km000', 'kn000', 'ko000', 'ko100', 'ko101', 'ko102', 'ko103', 
#     # 'ks000', 'ku000', 'ky000', 'la000', 'lb000', 'lg000', 'ln000', 'lo000', 'lt000', 'lv000', 'mg000', 'mi000', 'mk000', 
#     # 'ml000', 'mn000', 'mr000', 'ms000', 'my000', 'na000', 'nd000', 'ne000', 'nl000', 'nl100', 'no000', 'nv000', 'oc000', 
#     # 'om000', 'or000', 'pa000', 'pl000', 'ps000', 'pt000', 'pt100', 'pt101', 'pt102', 'pt103', 'qu000', 'rm000', 'rn000', 
#     # 'ro000', 'ru000', 'ru001', 'ru100', 'ru101', 'ru102', 'ru103', 'ru104', 'ru105', 'ru106', 'rw000', 'sa000', 'sc000', 
#     # 'sd000', 'sg000', 'sh000', 'si000', 'sk000', 'sl000', 'sm000', 'sn000', 'so000', 'sq000', 'sr000', 'st000', 'su000', 
#     # 'sv000', 'sw000', 'ta000', 'te000', 'tg000', 'th000', 'th100', 'ti000', 'tk000', 'tn000', 'to000', 'tr000', 'tr100', 
#     # 'ts000', 'tt000', 'ug000', 'uk000', 'uk100', 'ur000', 'uz000', 've000', 'vi000', 'vi100', 'vi101', 'vo000', 'wo000', 
#     # 'xh000', 'yi000', 'yo000', 'zh000', 'zu000']

#     configs = ['xh000', 'bo000','sm000','sh000','oc000','kl000','bo000','bi000','bh000']

#     # Execute in parallel
#     Parallel(n_jobs=1, backend="loky")(delayed(load_ds_and_make_transcription)(subset) for subset in configs)


import os
import pandas as pd
from datasets import load_dataset, Dataset
import string
from tqdm import tqdm
import logging
import soundfile as sf
from difflib import SequenceMatcher
from faster_whisper import WhisperModel
from joblib import Parallel, delayed
import time

# Set environment variables
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['CURL_CA_BUNDLE'] = ''

# Define supported languages
languages = [
    'af', 'am', 'ar', 'as', 'az', 'ba', 'be', 'bg', 'bn', 'bo', 'br', 'bs', 'ca', 'cs', 'cy', 'da', 'de', 'el', 'en', 
    'es', 'et', 'eu', 'fa', 'fi', 'fo', 'fr', 'gl', 'gu', 'ha', 'haw', 'he', 'hi', 'hr', 'ht', 'hu', 'hy', 'id', 'is', 
    'it', 'ja', 'jw', 'ka', 'kk', 'km', 'kn', 'ko', 'la', 'lb', 'ln', 'lo', 'lt', 'lv', 'mg', 'mi', 'mk', 'ml', 'mn', 
    'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'nn', 'no', 'oc', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'sa', 'sd', 'si', 'sk', 
    'sl', 'sn', 'so', 'sq', 'sr', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr', 'tt', 'uk', 'ur', 'uz', 
    'vi', 'yi', 'yo', 'zh', 'yue'
]

def preprocess_text(text):
    return text.strip().lower().translate(str.maketrans('', '', string.punctuation))

def save_csv(subset, results_ls_filtered):
    import csv
    with open(subset + '.csv', 'w', newline ='') as f:
        write = csv.writer(f)
        write.writerow(results_ls_filtered)

def load_model():
    model_size = "large-v3"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    return model

def making_transcription(subset, ds, model):
    language = subset[0:2]
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    correct_transcriptions = 0
    total = 0
    results_ls_filtered = []
    metadatas = []

    for i in tqdm(ds, desc=f"Processing Audio Samples for {subset}"):
        audio_dict = dict(i['audio'])
        audio_array = i['audio']['array']

        try:
            segments, _ = model.transcribe(audio_array, beam_size=5, language=None if language not in languages else language)
        except Exception as e:
            print(f"Error transcribing {i['id']}: {e}")
            continue

        total += 1
        expected_text = preprocess_text(i['text'])
        transcribed_text = ' '.join(segment.text for segment in segments)
        transcribed_text = preprocess_text(transcribed_text)

        if transcribed_text == expected_text:
            correct_transcriptions += 1
            results_ls_filtered.append(i['id'])
            metadatas.append({'id': i['id'], 'utt_id': i['utt_id'], 
                              'expected': i['text'], 'transcribed': transcribed_text, 'audio': audio_dict})
        
        accuracy = correct_transcriptions / total
        logging.info(f"Processed {total} samples for {subset}. Current accuracy: {accuracy:.2%}")

    final_accuracy = correct_transcriptions / total
    print(f"Final Accuracy for {subset}: {final_accuracy:.2%}")

    save_csv(subset, results_ls_filtered)

def load_ds_and_make_transcription(subset):
    model = load_model()
    try:
        ds = load_dataset('espnet/yodas', subset, split="train", streaming=True, trust_remote_code=True)
        print(f"Loaded dataset {subset}")
        making_transcription(subset, ds, model)
    except Exception as e:
        print(f"Error processing subset {subset}: {e}")

if __name__ == '__main__':
    configs = ['oc000','sm000','sh000','kl000']
    start_time = time.time()
    Parallel(n_jobs=3)(delayed(load_ds_and_make_transcription)(subset) for subset in configs)
    print("--- %s seconds ---" % (time.time() - start_time))
