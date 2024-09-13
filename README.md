Documentation for Running Inference using the Python Script

# 1\. System Requirements

To run the script efficiently, ensure your system meets the following specifications:  
\- GPU: CUDA-capable GPU is highly recommended (e.g., RTX 4090).  
\- Memory: At least 16GB of GPU memory is ideal for larger datasets.  
\- Python Libraries:  
\- PyTorch  
\- HuggingFace Datasets  
\- Faster Whisper (or Whisper)  
\- tqdm (for progress tracking)  
<br/>Ensure you have set up your environment with the necessary libraries before running the script.

# 2\. Setting Up the Dataset

There are two modes for dataset processing: local and streaming.  
<br/>\- **Local Mode**: You can download datasets locally, extract them, and pass them to the script.  
\- **Streaming Mode**: Directly stream the data without needing to download it, which is faster for large datasets but might have limits based on your internet speed and Hugging Face service.

\*If a dataset was streamed first and then run from local (extracting the files) the program will work, but not vice versa, as there is no way to get the **state_dict** of a local dataset.

# 3\. Running the Script

The script accepts multiple command-line arguments for different configurations.

## Basic Command Example (Inference)

**python dataproc_fw.py --sub=en100 --no-stream –whisper**

## Basic Command Example (Save to files)

**python save_todisk.py --sub=en100**

## Arguments

\- --sub: Specify the subset of data to run the inference on (e.g., en100 or es000).  
\- --no-stream: Runs the script in local mode. If omitted, it will default to streaming mode.  
\- --whisper: If you want to use the Whisper model for inference. This is the default mode.  
\- --fwhisper: Use the Faster Whisper model for faster inference.

Execution Notes:  
\- If running the script locally, ensure the dataset is properly saved and accessible on your system. Need to check in the code for the correct path of savedir, cache_dir.  
\- If using streaming, ensure a stable internet connection as the data will be fetched during runtime.
For **save_todisk** file:  
\- Make sure to use the variable diskdir (disk directory) and provide the external disk location (for example "E:/yodas/datasets").  
\- The chunk size when extracting the files is 200,000 which consumes almost 14GB of memory. In the case of running in slower computers, 
**chunk_size** variable should be reduced to consume less memory.

# 4\. Key Considerations

## Handling Model Speeds

\- Faster Whisper vs. Whisper: Depending on the model, inference speeds can vary. Based on tests, the script can run at:  
\- Faster Whisper: ~3 iteration per second.  
\- Distil Whisper: ~6 iterations per second, which is faster but only supports English.  
\- Consider reducing the beam size to speed up inference. Batch_size recommended is 16 but can be changed depending on characteristics of machine.

## Parallel Processing

\- Multiprocessing for subsets can be set up, but parallel execution on the same dataset is often not efficient. Limit to one job per subset for optimal performance.  
\- Running multiple jobs in parallel on the same GPU can lead to slower speeds or errors due to memory limitations. It is recommended to use more than one GPU to speed processes.

## CUDA Errors

\- When processing large datasets, CUDA errors may occur if the GPU runs out of memory. If this happens:  
\- Reduce the batch size or the dataset size.  
\- Use streaming mode to avoid loading large datasets into memory all at once.

# 5\. Saving Results

During inference, the results are saved progressively. The output is typically stored as a dictionary where each entry includes:  
\- ID: Identifier for the audio file.  
\- Original text: Ground truth transcription.  
\- Transcription: Model-predicted transcription.  
\- Audio file: File .wav of the audio is saved.

# 6\. Handling Errors

Common issues encountered during execution:  
1\. File Not Found Errors: If using Hugging Face streaming datasets, it can be an error connection after 10 seconds. Execution will keep going if this happens until successfull connection.  
2\. Memory Issues: For larger datasets, it is recommended to:  
\- Use streaming mode.  
\- Consider reducing the dataset size or switching to a smaller model like Distil Whisper.

# 7\. Improving Accuracy and Speed

\- Ensure that the correct language is set for the model. (--sub=xy###). The program automatically loads the correct model (distil for english and normal for everything else)

\- Use lower precision (FP16) or chunking to increase speed.

\- For large datasets, running the model in batches or subsets might speed up the process without overloading memory.