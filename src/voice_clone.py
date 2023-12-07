
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['COQUI_TOS_AGREED'] = '1'
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from src.log_helper import HandleLog

from huggingface_hub import snapshot_download
from src.temp_manager import TempFileManager
from typing import Any
from pathlib import Path
from basicsr.utils.download_util import load_file_from_url

logger = HandleLog()
class pre_VoiceCloner:

    def __init__(self) -> None:
        logger.info("Downloading xtts model")
        model_path = os.path.join('weights',"xtts_v2")
        snapshot_download(
            repo_id="coqui/XTTS-v2",
            local_dir=model_path,
            max_workers=8
        )
        logger.info("Loading xtts model...")
        config = XttsConfig()
        config.load_json("{}/config.json".format(model_path))
        model = Xtts.init_from_config(config=config)
        model.load_checkpoint(config, checkpoint_dir=model_path,use_deepspeed=False)
        if torch.cuda.is_available():
            model = model.cuda()
        self.model = model
        self.temp_manager = TempFileManager()

    def __call__(self, text, lang_code, speaker_wav,*args: Any, **kwds: Any) -> Any:
        logger.info("Computing speaker latents...")
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=speaker_wav)
        logger.info("xtts inferencing...")
        out = self.model.inference(text=text,
                                   language=lang_code,
                                   gpt_cond_latent=gpt_cond_latent,
                                   speaker_embedding=speaker_embedding,
                                   enable_text_splitting=True,
                                   num_beams=5)
        temp_file = self.temp_manager.create_temp_file(suffix='.wav').name
        
        torchaudio.save(temp_file, torch.tensor(out["wav"]).unsqueeze(0), 24000,bits_per_sample=16)
        return temp_file

from TTS.api import TTS

class VoiceCloner:

    def __init__(self, version_name="v2.0.3") -> None:
        self.temp_manager = TempFileManager()
        root_path = os.path.join('weights',f"xtts_{version_name}")
        config_path = load_file_from_url(url=f"https://hf-mirror.com/coqui/XTTS-v2/resolve/{version_name}/config.json?download=true",
                           model_dir=root_path,
                           file_name="config.json")
        load_file_from_url(url=f"https://hf-mirror.com/coqui/XTTS-v2/resolve/{version_name}/model.pth?download=true",
                           model_dir=root_path,
                           file_name="model.pth")
        load_file_from_url(url=f"https://hf-mirror.com/coqui/XTTS-v2/resolve/{version_name}/vocab.json?download=true",
                           model_dir=root_path,
                           file_name="vocab.json")
        load_file_from_url(url=f"https://hf-mirror.com/coqui/XTTS-v2/resolve/{version_name}/hash.md5?download=true",
                           model_dir=root_path,
                           file_name="hash.md5")
        # model_path = f"{root_path}/model.pth"
        # logger.info(f'model_path:{model_path}')
        self.tts = TTS(model_path=root_path,config_path=config_path,gpu=True)
    
    def __call__(self, text, lang_code, speaker_wav,speed=1.0,*args: Any, **kwds: Any) -> Any:
        temp_file = self.temp_manager.create_temp_file(suffix='.wav').name
        self.tts.tts_to_file(text=text,
                             language=lang_code,
                             speaker_wav=speaker_wav,
                             speed=speed,
                             file_path=temp_file)
        return temp_file