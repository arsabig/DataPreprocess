import time


from datasets import load_dataset, Dataset
from multiprocessing import Pool

def making_transcription(dataset):
    print(dataset['id'])

ds = load_dataset('espnet/yodas', 'rm000', split="train", cache_dir="E:/yodas/datasets", 
                  streaming=True, trust_remote_code=True)
# state_dict = iterable_dataset.state_dict()


if __name__ == '__main__':
    start_time = time.time()
    pool = Pool(2)
    pool.map(making_transcription,ds)
    print("--- %s seconds ---" % (time.time() - start_time))
    


