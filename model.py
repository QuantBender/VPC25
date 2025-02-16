#############################################################################
# YOUR ANONYMIZATION MODEL
# ---------------------
# Should be implemented in the 'anonymize' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# If you trained a machine learning model you can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY* <!>
############################################################################

import whisper
from TTS.api import TTS
from IPython.display import Audio
import torch
import numpy as np
import random

# Set random seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed
set_seed(0x0BADC0DE)

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
speech_to_test_model = whisper.load_model("large-v2") # large-v2, medium
test_to_speech_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

speaker_list = test_to_speech_model.speakers


def anonymize(input_audio_path): # <!> DO NOT ADD ANY OTHER ARGUMENTS <!>
    """
    anonymization algorithm

    Parameters
    ----------
    input_audio_path : str
        path to the source audio file in one ".wav" format.

    Returns
    -------
    audio : numpy.ndarray, shape (samples,), dtype=np.float32
        The anonymized audio signal as a 1D NumPy array of type `np.float32`, 
        which ensures compatibility with `soundfile.write()`.
    sr : int
        The sample rate of the processed audio.
    """
    speaker = str(input_audio_path).split('/')[2]
    speaker_id = int(speaker.replace('speaker', '')) - 1
    # print(f"<<<<<<<<<<<<<<<<<<<<<<<<<{speaker_id}>>>>>>>>>>>>>>>>>>>>>")

    # Read the source audio file
    #audio = Audio("/content/1272-128104-0000.wav")

    # Apply your anonymization algorithm
    # 01
    result = speech_to_test_model.transcribe(input_audio_path, language="en")
    transcribe = result["text"]
    # if transcribe.count(".") > 1:
    #     transcribe = ', '.join(transcribe.rsplit('.', maxsplit=1))  
    # transcribe = result["text"].replace('.', ', ')
    # transcribe = transcribe.replace('!', '  ')
    # transcribe = transcribe.replace('?', '?, ')
    # 02
    # test_to_speech_model.tts_to_file(transcribe, file_path="out.wav")
    audio_array = test_to_speech_model.tts(transcribe, language="en", speaker=speaker_list[speaker_id])

    # Output:
    audio = audio_array
    sr = 22050
    
    return audio, sr
