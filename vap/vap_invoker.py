
# definicion de parametros
from argparse import ArgumentParser
from os.path import basename

from vap.model import VapGPT, VapConfig, load_older_state_dict
from vap.audio import load_waveform
from vap.utils import (
    batch_to_device,
    everything_deterministic,
    tensor_dict_to_json,
    write_json,
)
from vap.plot_utils import plot_stereo
import torch
import torchaudio
import io
import io

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import torch
import torchaudio
import math
import torchaudio.functional as AF
import time



def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-a",
        "--audio",
        type=str,
        help="Path to waveform",
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        default=None,
        help="Path to waveform",
    )
    parser.add_argument(
        "-sd",
        "--state_dict",
        type=str,
        default="example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt",
        help="Path to state_dict",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model",
    )
    parser, _ = VapConfig.add_argparse_args(parser)
    parser.add_argument(
        "--chunk",
        action="store_true",
        help="Process the audio in chunks (longer > 164s on 24Gb GPU audio)",
    )
    parser.add_argument(
        "--chunk_time",
        type=float,
        default=30,
        help="Duration of each chunk processed by model",
    )
    parser.add_argument(
        "--step_time",
        type=float,
        default=5,
        help="Increment to process in a step",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Visualize output (matplotlib)"
    )
    args = parser.parse_args()

    conf = VapConfig.args_to_conf(args)
    return args, conf


class VapInvoker():
    def __init__(self, state_dict= 'example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt'):
        args, conf = get_args()
        print("From state-dict: ", state_dict)
        self.model = VapGPT(conf)
        sd = torch.load(state_dict)
        self.model.load_state_dict(sd)

        self.device = "cpu"
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            self.device = "cuda"
        self.model = self.model.eval()

        self.model_sample_rate = 16000
        

    def invoke(self, buffer, audio_sample_rate = 48000):
        # Convertir el buffer a bytes
        buf = io.BytesIO()
        buffer.export(buf, format="wav")  # Exportar el buffer como WAV en memoria

        # Cargar el buffer como un tensor en PyTorch
        buf.seek(0)  # Volver al inicio del buffer para lectura
        waveform, sample_rate = torchaudio.load(buf)


        waveform = AF.resample(waveform, orig_freq=audio_sample_rate, new_freq=self.model_sample_rate)
        # sr = sample_rate

        # waveform, _ = load_waveform(audio_path, sample_rate=model.sample_rate)
        # duration = round(waveform.shape[-1] / model.sample_rate)
        if waveform.shape[0] == 1:
            waveform = torch.cat((waveform, torch.zeros_like(waveform)))
        waveform = waveform.unsqueeze(0)



        if torch.cuda.is_available():
            waveform = waveform.to("cuda")


        # start_time = time.time()  # Guardar el tiempo de inicio
        out = self.model.probs(waveform)
        # print(f"Tiempo de ejecución invoker: {time.time() - start_time} segundos")

        out = batch_to_device(out, "cpu")  # to cpu for plot/save
        return out
    
    