import torch
import torch.multiprocessing as mp
import gc
import sys
import string
import logging
from tabnanny import verbose
from networkx import johnson, node_disjoint_paths
from datasets import load_dataset, Dataset
from tqdm import tqdm  # For progress bar
from difflib import SequenceMatcher
from difflib import SequenceMatcher
from faster_whisper import WhisperModel
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from multiprocessing import get_context
from logging import getLogger


LOG = getLogger(__name__)
LOG.setLevel(logging.WARNING)
# from utils import get_total_gpu_memory
MODEL_MEMORY = 5000
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
    while True:
        try:
            ds = load_dataset('espnet/yodas', subset, split="train", streaming=True, trust_remote_code=True, )
            print('Read this >>>' , subset)
            # print(ds)
            making_transcription(subset, ds, lm)
            # this STREAMING loading will finish quickly, can load only one by one
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

def making_transcription(subset, ds, model):
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    language = subset[0:2] if subset[0:2] in languages else None
    correct_transcriptions = 0
    total = 0
    transcribed_text = ''     #needed if no segments come
    results_ls_filtered = []
    metadatas = []

    for i in tqdm(ds, desc=f"Processing {subset} Audio Samples"):
        # audio_dict = dict(i['audio']) #somehow i['audio'] gets empty {} after used in pipeline
        audio_array = i['audio']['array']

        with torch.no_grad():
            try:
                results_ls_filtered.append(i['id'])
                # segments, _ = model.transcribe(audio_array, beam_size=1, language=language)
            except Exception as e: print(e)

            total += 1

            save_temp(subset, results_ls_filtered)
            
            accuracy = 0# correct_transcriptions / total

            torch.cuda.empty_cache()

            
    final_accuracy = correct_transcriptions / total


    save_csv(subset, results_ls_filtered, correct_transcriptions, total, final_accuracy)
    # save_ds(subset, metadatas_ds) # SAVE DICTIONARY TO DISK, REQUIRES SPACE

def save_ds(subset, metadatas_ds):

def save_csv(subset, results_ls_filtered, ct, tot, acc):

def save_temp(subset, results_ls_filtered):


def open_csv():

def process_subset():
    configs = ['en100','es102','es104','es105','es103','en108','en126','en107','en119','en106','es101','en122','en121','en120','en103',
'en104','es100','en116','en101','en112','en113','ru104','ru102','en111','en118','en102','en117','en124','en109','en114','en115',
'en125','en110','ru103','ru100','ru101','en105','en123','fr102','ru105','fr100','fr101','en004','ko101','pt100','pt102','en002',
'pt101','en001','ko100','ko102','en003','en000','de100','de101','it100','ru000','vi100','id100','it101','vi101','tr100','en005',
'es000','id101','pt103','ru106','fr103','de000','ja100','ko000','es106','ko103','nl100','fr000','en127','pt000','it000','id000',
'ja000','de102','vi000','pl000','ru001','tr000','th000','hi000','uk100','nl000','uk000','zh000','ar000','fi000','hu000','cs000',
'iw000','no000','sv000','el000','ro000','ca000','ta000','be000','bg000','fa000','sk000','ms000','da000','bn000','ka000','hr000',
'sl000','ur000','eu000','lt000','sr000','et000','ky000','ml000','eo000','gl000','bs000','la000','mr000','te000','mk000','uz000',
'cy000','is000','si000','km000','az000','kk000','sq000','so000','hi100','lv000','kn000','my000','ne000','mn000','gu000','ku000',
'sw000','th100','hy000','pa000','ga000','mi000','jv000','ht000','ps000','am000','af000','qu000','bo000','br000','rw000','as000',
'or000','ab000','sa000','ti000','yo000','tg000','sh000','ak000','lo000','vo000','rm000','ln000','fo000','gn000','aa000','mg000',
'oc000','om000','zu000','ie000','xh000','tn000','lb000','ha000','sm000','ug000','ig000','ia000','yi000','wo000','sd000','tk000',
'fy000','dz000','iu000','ho000','tt000','co000','ee000','su000','na000','ff000','ay000','ba000','gd000','fj000','ks000','sn000',
'bh000','bi000','sc000','cr000','kl000','ik000','rn000','lg000','ve000','st000','nv000','bm000','nd000','ts000','ki000','to000',
'sg000']
    
    Parallel(n_jobs=2, backend="multiprocessing")(delayed(load_ds)(subset) for subset in configs)


if __name__ == '__main__':
    process_subset()