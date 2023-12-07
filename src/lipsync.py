import os
import cv2
import numpy as np
import torch
from typing import Any
from tqdm import tqdm
import subprocess
import platform
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
from src.log_helper import HandleLog
from src.third_part.wav2lip import face_detection
from src.third_part.wav2lip.models import Wav2Lip
from src.third_part.wav2lip.audio import *
from basicsr.utils.download_util import load_file_from_url
from src.temp_manager import TempFileManager
logger = HandleLog()

class LipSync:
    def __init__(self,model_name) -> None:
        self.model_name = model_name
        self.img_size = 96
        self.static = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.face_det_batch_size = 16
        self.wav2lip_batch_size = 16
        self.mel_step_size = 16
        self.pads = [0,20,0,0]
        self.nosmooth = True
        self.box = [-1, -1, -1, -1]
        self.fps = 25
        self.resize_factor = 2
        self.rotate = False
        self.crop = [0, -1, 0, -1]
        logger.info('Using {} for inference.'.format(self.device))

        load_file_from_url(url="https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
                           model_dir="src/third_part/wav2lip/face_detection/detection/sfd",
                           file_name="s3fd.pth")
        load_file_from_url(url="https://hf-mirror.com/MarjorieSaul/wav2lip_sd_models/resolve/main/wav2lip.pth?download=true",
                           model_dir="weights",
                           file_name="wav2lip.pth")
        load_file_from_url(url="https://hf-mirror.com/MarjorieSaul/wav2lip_sd_models/resolve/main/wav2lip_gan.pth?download=true",
                           model_dir="weights",
                           file_name="wav2lip_gan.pth")
        self.tmp_manager = TempFileManager()
        

    def __call__(self, face,audio,outfile,voice,*args: Any, **kwds: Any) -> Any:
        if os.path.isfile(face) and face.split('.')[1] in ['jpg', 'png', 'jpeg']:
            self.static = True
        if not os.path.isfile(face):
            raise ValueError('face argument must be a valid path to video/image file')
        elif face.split('.')[1] in ['jpg', 'png', 'jpeg']:
            full_frames = [cv2.imread(face)]
            fps = self.fps
        else:
            video_stream = cv2.VideoCapture(face)
            fps = video_stream.get(cv2.CAP_PROP_FPS)
            logger.info('Reading video frames...')

            full_frames = []
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                if self.resize_factor > 1:
                    frame = cv2.resize(frame, (frame.shape[1]//self.resize_factor, frame.shape[0]//self.resize_factor))

                if self.rotate:
                    frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
                
                y1, y2, x1, x2 = self.crop
                if x2 == -1: x2 = frame.shape[1]
                if y2 == -1: y2 = frame.shape[0]
                frame = frame[y1:y2, x1:x2]
                full_frames.append(frame)

        logger.info("Number of frames available for inference: "+str(len(full_frames)))

        assert audio.endswith('.wav'),"audio file shoud end with .wav"

        wav = load_wav(audio, sr=16000)
        mel = melspectrogram(wav)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')
        
        mel_chunks = []
        mel_idx_multiplier = 80./fps
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + self.mel_step_size])
            i += 1
        
        logger.info("Length of mel chunks: {}".format(len(mel_chunks)))

        full_frames = full_frames[:len(mel_chunks)]

        batch_size = self.wav2lip_batch_size

        gen = self.datagen(full_frames.copy(), mel_chunks)
        while 1:
            try:
                for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
                                                                                total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
                    if i == 0:
                        model = self.load_model()
                        logger.info("Model loaded")
                        frame_h, frame_w = full_frames[0].shape[:-1]
                        temp_file = self.tmp_manager.create_temp_file(suffix='.avi').name
                        out = cv2.VideoWriter(temp_file, 
									cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
                    img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
                    mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)

                    with torch.no_grad():
                        pred = model(mel_batch, img_batch)
                    pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
                    for p, f, c in zip(pred, frames, coords):
                        y1, y2, x1, x2 = c
                        try:
                            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                            f[y1:y2, x1:x2] = p
                        except cv2.error:
                            pass
                        out.write(f)
                out.release()
            except RuntimeError:
                if batch_size == 1: 
                    raise RuntimeError('Image too big to run wav2lip on GPU. Please use the --resize_factor argument')
                batch_size //= 2
                continue
            break
        command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(voice, temp_file, outfile)
        subprocess.call(command, shell=platform.system() != 'Windows')
                
    def load_model(self):
        model = Wav2Lip()
        logger.info("Load checkpoint from: {}".format(self.model_name))
        checkpoint = self._load()
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)
        model = model.to(self.device)
        return model.eval()

    def _load(self):
        checkpoint_path = "weights/{}.pth".format(self.model_name)
        if self.device == 'cuda':
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path,
                                    map_location=lambda storage, loc: storage)
        return checkpoint
    

    def datagen(self,frames, mels):
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if self.box[0] == -1:
            if not self.static:
                face_det_results = self.face_detect(frames) # BGR2RGB for CNN face detection
            else:
                face_det_results = self.face_detect([frames[0]])
        else:
            logger.info('Using the specified bounding box instead of face detection...')
            y1, y2, x1, x2 = self.box
            face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
        
        for i, m in enumerate(mels):
            idx = 0 if self.static else i%len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx].copy()

            try:
                face = cv2.resize(face, (self.img_size, self.img_size))
            except cv2.error:
                face = np.zeros((10, 10,3), np.uint8)
                face = cv2.resize(face, (self.img_size, self.img_size))
            
            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= self.wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, self.img_size//2:] = 0
                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
        
        
        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
            img_masked = img_batch.copy()
            img_masked[:, self.img_size//2:] = 0
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            yield img_batch, mel_batch, frame_batch, coords_batch


    def face_detect(self,images):
        detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                                flip_input=False,device=self.device
                                                )
        batch_size = self.face_det_batch_size
        while 1:
            predictions = []
            try:
                for i in tqdm(range(0,len(images),batch_size)):
                    predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
            except RuntimeError:
                if batch_size == 1: 
                    raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
                batch_size //= 2
                logger.warning('Recovering from OOM error; New batch size: {}'.format(batch_size))
                continue
            break
        results = []
        pady1, pady2, padx1, padx2 = self.pads
        for rect, image in zip(predictions, images):
            if rect is None:
                rect = (0,20,0,0)
            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)
            results.append([x1,y1,x2,y2])
        boxes = np.array(results)
        if not self.nosmooth: boxes = self.get_smoothened_boxes(boxes, T=5)
        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]
        import gc; gc.collect(); torch.cuda.empty_cache();del detector
        return results

    def get_smoothened_boxes(self,boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
            return boxes
    