import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
from torch import torch
from logging import getLogger
from joblib import Parallel, cpu_count, delayed
import numpy as np
from tqdm import tqdm
from multiprocessing import cpu_count
import glob
import argparse
# from utils import get_total_gpu_memory
import logging
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
import time
# import torchaudio.functional as F
MODEL_MEMORY = 5000
LOG = getLogger(__name__)
LOG.setLevel(logging.WARNING)
def build_from_path(in_dir, out_dir, args, n_jobs=None):
    args.out_dir = out_dir
    # if n_jobs is None:
    #     # add cpu_count() to avoid SIGKILL
    #     memory = torch.cuda.mem_get_info(device='cuda:0')[0]//1024 ** 2
    #     n_jobs = min(
    #         max(
    #             memory
    #             // MODEL_MEMORY
    #             if memory is not None
    #             else 1,
    #             1,
    #         ),
    #         cpu_count(),
    #     )
    #     LOG.info(f"n_jobs automatically set to {n_jobs}, memory: {memory} MiB")
    n_jobs = 2
    wavfile_paths = glob.glob(os.path.join(in_dir, '*.wav'))
    wavfile_paths = sorted(wavfile_paths)
    n_jobs = min(len(wavfile_paths) // 16 + 1, n_jobs)
    filepath_chunks = np.array_split(wavfile_paths, n_jobs)
    gpu_list = [int(n) for n in args.n_gpu.split(",")]
    number_of_gpu = len(gpu_list)
    Parallel(n_jobs=n_jobs)(
        delayed(_process_batch)(
            filepaths=chunk,
            pbar_position=pbar_position,
            args=args,
            n_gpu=gpu_list[pbar_position%number_of_gpu])
        for (pbar_position, chunk) in enumerate(filepath_chunks)
    )
def _process_batch(filepaths, pbar_position, args, n_gpu, **kwargs):
    model = WhisperModel("large-v3", device='cuda', compute_type="float16")
    for filepath in tqdm(filepaths, position=pbar_position):
        _compute_sr(filepath, model, args)
def _get_encoder_output(audio, cmodel):
    sampling_rate = cmodel.feature_extractor.sampling_rate
    if not isinstance(audio, np.ndarray):
        audio = decode_audio(audio, sampling_rate=sampling_rate)
    duration = audio.shape[0] / sampling_rate
    features = cmodel.feature_extractor(audio)
    segment = features[:, : cmodel.feature_extractor.nb_max_frames]
    encoder_output = cmodel.encode(segment)
    return encoder_output, duration
def _compute_sr(filename, cmodel, args):
    basename = os.path.basename(filename)
    folder_name = basename.replace(".wav",'')
    folder_path = os.path.join(args.out_dir, folder_name)
    ppgPath = os.path.join(folder_path, basename.replace(".wav", "ppg.npy"))
    # if not os.path.isdir(folder_path):
    os.makedirs(folder_path, exist_ok=True)
    with torch.no_grad():
        ppg, duration = _get_encoder_output(filename, cmodel)
        # print(ppg.shape)
        np.save(ppgPath, torch.tensor(ppg).numpy()[0][:math.ceil(duration*50)], allow_pickle=False)
        # np.save(ppgPath, torch.tensor(ppg).numpy()[0], allow_pickle=False)
        torch.cuda.empty_cache()
if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=str, default='0', help="")
    parser.add_argument('--in_dir', type=str,
                        default=r'C:\Users\hudso\Downloads\00000000')
    parser.add_argument('--out_dir_root', type=str,
                        default=r'C:\Users\hudso\Downloads\qui')
    parser.add_argument('--symbol', type=str, default=None)
    args = parser.parse_args()
    sub_folder_list = os.listdir(args.in_dir)
    sub_folder_list.sort()
    for spk in sub_folder_list:
        print("Preprocessing {} ...".format(spk))
        in_dir = os.path.join(args.in_dir, spk)
        # if args.symbol is not None:
        #     if args.symbol not in spk:
        #         continue
        # if not os.path.isdir(in_dir):
        #     continue
        # if os.path.isdir(os.path.join(args.out_dir_root, spk)):
        #     continue
        build_from_path(in_dir, os.path.join(args.out_dir_root, spk), args)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("DONE!")
    sys.exit(0)