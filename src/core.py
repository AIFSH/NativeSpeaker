import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import soundfile as sf
from typing import Any
from tqdm import tqdm
from src.log_helper import HandleLog
from moviepy.editor import VideoFileClip,concatenate_videoclips
from pathlib import Path
from pydub import AudioSegment
from src.audio_bgm_split import AudioProcess
from src.voice_clone import VoiceCloner
from src.temp_manager import TempFileManager
from src.translator import Translator
from src.lipsync import LipSync
from src.upscale import Upscale
from src.nfsw import analyse_video
from src.third_part.whisperx import load_model,load_audio,DiarizationPipeline

logger = HandleLog()

class Core:
    def __init__(self, args) -> None:
        cur_path = os.path.dirname(os.path.realpath(__file__))  # current path
        self.weights_path = os.path.join(os.path.dirname(cur_path), 'weights')  # weights_path to save model
        if not os.path.exists(self.weights_path): os.mkdir(self.weights_path)  # 

        self.input_file = args.input_file_path
        self.output_file = args.output_file_path
        self.lang_code = args.lang_code
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hf_token = args.hf_token
        self.temp_manager = TempFileManager()
        self.translotor = Translator()
        self.model_name = args.model_name
        self.xt_version_name = args.xt_version_name

        if analyse_video(args.input_file_path):
            raise("sorry! nativespeaker is not for you")

        
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        logger.critical("[Step 1] Moviepy split voice and frames from video")
        org_voice_path = os.path.join(Path(self.input_file).parent, "org_voice.wav")
        org_video_clip = VideoFileClip(self.input_file)
        org_video_clip.audio.write_audiofile(org_voice_path,codec='pcm_s16le')
        logger.info("save original voice in {}".format(org_voice_path))

        logger.critical("[Step 2] H5 Split vocal and bgm from voice")
        audio_process = AudioProcess(15)
        vocal_file, bgm_file = audio_process.split(org_voice_path)

        logger.critical("[Step 3] whisperx from speech to text")
        whispher_segments, src_lang_code, speakers_wav = self.speech_to_text(vocal_file)

        logger.critical("[Step 4] translate,text to speech,video and voice_cloned aligment")
        vocal_cloned_audio = AudioSegment.silent(0)
        bgm_audio_extend = AudioSegment.silent(0)
        video_extend_list = []

        org_vocal = AudioSegment.from_file(vocal_file)
        bgm_audio = AudioSegment.from_file(bgm_file)

        seg_len = len(whispher_segments)
        cloner = VoiceCloner(self.xt_version_name)
        root_path = Path(self.output_file).parent
        zimu_txt = os.path.join(root_path,"zimu.txt")
        for i, segment in tqdm(enumerate(whispher_segments), desc="voice cloning", total=seg_len):
            start = segment['start'] * 1000
            end = segment['end'] * 1000
            text_list = segment['text_list']
            if i == 0:
                vocal_cloned_audio += org_vocal[:start]
                bgm_audio_extend += bgm_audio[:start]

                video_extend_list.append(org_video_clip.subclip(0, start / 1000))
            
            total_cloned_vocal = AudioSegment.silent(0)
            if len(text_list) > 0:
                for src_text in text_list:
                    dst_text = self.translotor(src_text,src_lang_code,self.lang_code)

                    with open(zimu_txt,mode="a",encoding="utf-8",newline="\n") as w:
                        w.write(src_text)
                        w.write("\n")
                        w.write(dst_text)
                        w.write("\n")

                    cloned_vocal_path = cloner(text=dst_text,
                                            lang_code=self.lang_code,
                                            speaker_wav=[segment['wav'],speakers_wav[segment['speaker']]])
                    cloned_vocal = AudioSegment.from_file(cloned_vocal_path)

                    total_cloned_vocal += cloned_vocal
            else:
                logger.info(f'no sound there')
                total_cloned_vocal = AudioSegment.silent(end - start)
            
            vocal_cloned_audio += total_cloned_vocal
            tmp_bgm_audio = bgm_audio[start:end]
            bgm_audio_extend += self.bgm_map_vocal(tmp_bgm_audio, total_cloned_vocal)

            tmp_video_clip = org_video_clip.subclip(start/1000, end/1000)
            tmp_video_clip = self.video_map_vocal(tmp_video_clip, total_cloned_vocal)
            video_extend_list.append(tmp_video_clip)

            if i < seg_len - 1:
                # duration
                vocal_cloned_audio += org_vocal[end:whispher_segments[i+1]['start']*1000]
                bgm_audio_extend += bgm_audio[end:whispher_segments[i+1]['start']*1000]
                video_extend_list.append(org_video_clip.subclip(end/1000, whispher_segments[i+1]['start']))

            if i == seg_len - 1:
                # duration
                vocal_cloned_audio += org_vocal[end:]
                bgm_audio_extend += bgm_audio[end:]
                video_extend_list.append(org_video_clip.subclip(end/1000))

        
        vocal_cloned_path = os.path.join(root_path,"vocal_cloned.wav")
        vocal_cloned_audio.export(vocal_cloned_path, format="wav")
        logger.info("vocal_cloned.wav saved in {}, you can check it".format(root_path))

        bgm_extend_path = os.path.join(root_path,"bgm_extend.wav")
        bgm_audio_extend.export(bgm_extend_path, format="wav")
        logger.info("bgm_extend.wav saved in {}, you can check it".format(root_path))

        voice_cloned_path = os.path.join(root_path,"voice_cloned.wav")
        self.combie_audio(vocal_cloned_audio,bgm_audio_extend,voice_cloned_path)
        logger.info("voice_cloned.wav saved in {}, you can check it".format(root_path))


        video_extend_path = os.path.join(root_path,"video_extend.mp4")
        video_extended = concatenate_videoclips(video_extend_list)
        video_extended.write_videofile(video_extend_path,fps=25,audio=False)
        logger.info("video_extend.mp4 saved in {}, you can check it".format(root_path))
        # delete model if low on GPU resources
        import gc; gc.collect(); torch.cuda.empty_cache(); del cloner

        logger.critical("[Step 5] Wav2Lip by vocal_cloned.wav and video_extend.mp4")
        lipsync = LipSync(self.model_name)
        lipsync(video_extend_path,vocal_cloned_path,self.output_file,voice_cloned_path)

        logger.critical("[Step 6] Upscale output video last step")
        
        upscaler = Upscale()
        upscale_workplace_path = os.path.join(root_path,"upscale_workplace")
        upscaler(input_path = self.output_file,output_path=upscale_workplace_path,audio=voice_cloned_path)
        
        self.temp_manager.cleanup()


    def combie_audio(self,vocal_audio:AudioSegment,bgm_audio:AudioSegment,file_path):
        new_audio = vocal_audio.overlay(bgm_audio)
        new_audio.export(file_path, format="wav")

    def bgm_map_vocal(self,bgm_audio:AudioSegment,vocal_audio:AudioSegment):
        audio_duration = vocal_audio.duration_seconds

        ratio = audio_duration / bgm_audio.duration_seconds
        print("audio.duration_seconds /  bgm.duration_seconds = {}".format(ratio))
        tmp_bgm_path = self.temp_manager.create_temp_file(suffix='.wav').name
        bgm_audio.export(tmp_bgm_path, format="wav")
        bgm_path = self.temp_manager.create_temp_file(suffix='.wav').name
        y,sr = sf.read(tmp_bgm_path)
        sf.write(bgm_path,y,int(sr*ratio))
        bgm_extended = AudioSegment.from_file(bgm_path)
        return bgm_extended[:audio_duration * 1000]

    def video_map_vocal(self,vido_clip:VideoFileClip,vocal_audio:AudioSegment):
        audio_duration = vocal_audio.duration_seconds
        
        video_duration = vido_clip.duration
        
        ratio = video_duration / audio_duration 
        print("video_duration / audio_duration  =ratio:{}".format(ratio))
        
        new_video = vido_clip.fl_time(lambda t:  ratio*t,apply_to=['mask', 'audio'])
        new_video1 = new_video.set_duration(audio_duration)
        new_video2 = new_video1.set_fps(new_video1.fps / video_duration * audio_duration)
        return new_video2.subclip(0,audio_duration)

    def speech_to_text(self, vocal_file):
        vocal_audio = load_audio(vocal_file)
        batch_size = 32
        # 1. Assign speaker labels
        diarize_model = DiarizationPipeline(use_auth_token=self.hf_token, device=self.device)

        # add min/max number of speakers if known
        diarize_segments  = diarize_model(vocal_audio, min_speakers=1, max_speakers=5)
        # logger.info(diarize_segments)
        speakers_list = diarize_segments.iloc[:,2].values.tolist()
        start_list = diarize_segments.iloc[:,3].values.tolist()
        end_list = diarize_segments.iloc[:,4].values.tolist()
        # drop_duplit
        speakers = list(set(speakers_list))
        speakers_audios_dict = { key: AudioSegment.silent(0) for key in speakers}
        org_audio = AudioSegment.from_file(vocal_file)

        whispher_segments = []

        whisper = load_model("large-v3", device=self.device,download_root=self.weights_path)
        
        # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
        import gc; gc.collect(); torch.cuda.empty_cache(); del diarize_model

        # logger.debug("Before alignment:".format(result["segments"]))
        
        lang_code = whisper.detect_language(vocal_audio)
        
        for start,end, speaker in tqdm(zip(start_list,end_list,speakers_list),\
                                       desc="Transcribing...",total=len(start_list)):
            clip_audio = org_audio[start*1000:end*1000]
            temp_file = self.temp_manager.create_temp_file(suffix='.wav')
            speakers_audios_dict[speaker] += clip_audio
            speakers_audios_dict[speaker] += AudioSegment.silent(1000)
            clip_audio.export(temp_file,format="wav")
            clip_audio_tmp = load_audio(temp_file.name)
            while 1:
                
                try:
                    result = whisper.transcribe(clip_audio_tmp, batch_size=batch_size,language=lang_code)
                except RuntimeError:
                    if batch_size == 1:
                        raise("Out of memory error!!! try to short audio time or big vam")
                    batch_size //=2
                    logger.warning("Out of memory error, redefine batch_size={}".format(batch_size))
                    continue
                break
            
            text_list = []
            for segment in result["segments"]:
                text = segment['text']
                if len(text) > 0:
                    text_list.append(text)
            
            whispher_segments.append({"speaker":speaker,"start":start, "end": end, \
                                      "text_list": text_list, "wav":temp_file.name})


        # delete model if low on GPU resources
        import gc; gc.collect(); torch.cuda.empty_cache(); del whisper
        
        speakers_wav = {}
        for key in speakers_audios_dict.keys():
            temp_file = self.temp_manager.create_temp_file(suffix='.wav')
            speakers_audios_dict[key].export(temp_file,format="wav")
            speakers_wav[key] = temp_file.name
        logger.info(speakers_wav)
        return whispher_segments, lang_code, speakers_wav
    










