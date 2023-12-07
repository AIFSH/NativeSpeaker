import os
import logging

logger = logging.getLogger(__name__)

import librosa
import numpy as np
import soundfile as sf
import torch

from src.third_part.uvr5_pack.lib_v5 import nets_61968KB as Nets
from src.third_part.uvr5_pack.lib_v5 import spec_utils
from src.third_part.uvr5_pack.lib_v5.model_param_init import ModelParameters
from src.third_part.uvr5_pack.utils import inference
from src.temp_manager import TempFileManager
from basicsr.utils.download_util import load_file_from_url


class AudioProcess:
    def __init__(self, agg, is_half=False, tta=False):

        # model_path = os.path.join('weights', 'HP5-主旋律人声vocals+其他instrumentals.pth')
        model_path = load_file_from_url(url="https://hf-mirror.com/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP5-%E4%B8%BB%E6%97%8B%E5%BE%8B%E4%BA%BA%E5%A3%B0vocals%2B%E5%85%B6%E4%BB%96instrumentals.pth?download=true", 
                                        model_dir='weights', progress=True, file_name="HP5-主旋律人声vocals+其他instrumentals.pth")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data = {
            # Processing Options
            "postprocess": False,
            "tta": tta,
            # Constants
            "window_size": 512,
            "agg": agg,
            "high_end_process": "mirroring",
        }
        mp = ModelParameters("src/third_part/uvr5_pack/lib_v5/modelparams/4band_v2.json")
        model = Nets.CascadedASPPNet(mp.param["bins"] * 2)
        cpk = torch.load(model_path, map_location="cpu")
        model.load_state_dict(cpk)
        model.eval()
        if is_half:
            model = model.half().to(self.device)
        else:
            model = model.to(self.device)

        self.mp = mp
        self.model = model

    def split(self, music_file):
        
        X_wave, y_wave, X_spec_s, y_spec_s = {}, {}, {}, {}
        bands_n = len(self.mp.param["band"])
        # print(bands_n)
        for d in range(bands_n, 0, -1):
            bp = self.mp.param["band"][d]
            if d == bands_n:  # high-end band
                (
                    X_wave[d],
                    _,
                ) = librosa.core.load(  # 理论上librosa读取可能对某些音频有bug，应该上ffmpeg读取，但是太麻烦了弃坑
                    path=music_file,
                    sr=bp["sr"],
                    mono=False,
                    dtype=np.float32,
                    res_type=bp["res_type"],
                )
                if X_wave[d].ndim == 1:
                    X_wave[d] = np.asfortranarray([X_wave[d], X_wave[d]])
            else:  # lower bands
                X_wave[d] = librosa.core.resample(
                    y=X_wave[d + 1],
                    orig_sr=self.mp.param["band"][d + 1]["sr"],
                    target_sr=bp["sr"],
                    res_type=bp["res_type"],
                )
            # Stft of wave source
            X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(
                X_wave[d],
                bp["hl"],
                bp["n_fft"],
                self.mp.param["mid_side"],
                self.mp.param["mid_side_b2"],
                self.mp.param["reverse"],
            )
            # pdb.set_trace()
            if d == bands_n and self.data["high_end_process"] != "none":
                input_high_end_h = (bp["n_fft"] // 2 - bp["crop_stop"]) + (
                    self.mp.param["pre_filter_stop"] - self.mp.param["pre_filter_start"]
                )
                input_high_end = X_spec_s[d][
                    :, bp["n_fft"] // 2 - input_high_end_h : bp["n_fft"] // 2, :
                ]

        X_spec_m = spec_utils.combine_spectrograms(X_spec_s, self.mp)
        aggresive_set = float(self.data["agg"] / 100)
        aggressiveness = {
            "value": aggresive_set,
            "split_bin": self.mp.param["band"][1]["crop_stop"],
        }
        with torch.no_grad():
            pred, X_mag, X_phase = inference(
                X_spec_m, self.device, self.model, aggressiveness, self.data
            )
        # Postprocess
        if self.data["postprocess"]:
            pred_inv = np.clip(X_mag - pred, 0, np.inf)
            pred = spec_utils.mask_silence(pred, pred_inv)
        y_spec_m = pred * X_phase
        v_spec_m = X_spec_m - y_spec_m

        
        if self.data["high_end_process"].startswith("mirroring"):
            input_high_end_y = spec_utils.mirroring(
                self.data["high_end_process"], y_spec_m, input_high_end, self.mp
            )
            wav_instrument = spec_utils.cmb_spectrogram_to_wave(
                y_spec_m, self.mp, input_high_end_h, input_high_end_y
            )
            
            input_high_end_v = spec_utils.mirroring(
                    self.data["high_end_process"], v_spec_m, input_high_end, self.mp
                )
            wav_vocals = spec_utils.cmb_spectrogram_to_wave(
                v_spec_m, self.mp, input_high_end_h, input_high_end_v
            )
            
        else:
            wav_instrument = spec_utils.cmb_spectrogram_to_wave(y_spec_m, self.mp)
            wav_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, self.mp)
            
        logger.info("vocal and instruments split done")
        
        temp_manager = TempFileManager()
        voice_temp_file = temp_manager.create_temp_file(suffix='.wav')
        noise_temp_file = temp_manager.create_temp_file(suffix='.wav')
        
        sf.write(
            voice_temp_file,
            (np.array(wav_vocals) * 32768).astype("int16"),
            self.mp.param["sr"],
        )
        sf.write(
            noise_temp_file,
            (np.array(wav_instrument) * 32768).astype("int16"),
            self.mp.param["sr"],
        )
        return voice_temp_file.name, noise_temp_file.name

