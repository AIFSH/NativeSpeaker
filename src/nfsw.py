import cv2,numpy
import torch
from tqdm import tqdm
import onnxruntime
from functools import lru_cache
from typing import Optional,Any,List
from basicsr.utils.download_util import load_file_from_url

MAX_PROBABILITY = 0.80
MAX_RATE = 5
STREAM_COUNTER = 0 


@lru_cache(maxsize = None)
def analyse_video(video_path : str) -> bool:
	video_frame_total = count_video_frame_total(video_path)
	fps = detect_fps(video_path)
	frame_range = range( 0, video_frame_total)
	rate = 0.0
	counter = 0
	with tqdm(total = len(frame_range), desc = 'video content analysing', unit = 'frame', ascii = ' =') as progress:
		for frame_number in frame_range:
			if frame_number % int(fps) == 0:
				frame = get_video_frame(video_path, frame_number)
				if analyse_frame(frame):
					counter += 1
			rate = counter * int(fps) / len(frame_range) * 100
			progress.update()
			progress.set_postfix(rate = rate)
	return rate > MAX_RATE

def count_video_frame_total(video_path : str) -> int:
	if video_path:
		video_capture = cv2.VideoCapture(video_path)
		if video_capture.isOpened():
			video_frame_total = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
			video_capture.release()
			return video_frame_total
	return 0

def detect_fps(video_path : str) -> Optional[float]:
	if video_path:
		video_capture = cv2.VideoCapture(video_path)
		if video_capture.isOpened():
			return video_capture.get(cv2.CAP_PROP_FPS)
	return None

def get_video_frame(video_path : str, frame_number : int = 0) -> Any:
	if video_path:
		video_capture = cv2.VideoCapture(video_path)
		if video_capture.isOpened():
			frame_total = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
			video_capture.set(cv2.CAP_PROP_POS_FRAMES, min(frame_total, frame_number - 1))
			has_frame, frame = video_capture.read()
			video_capture.release()
			if has_frame:
				return frame
	return None

def analyse_frame(frame) -> bool:
	model_path = load_file_from_url(url="https://github.com/facefusion/facefusion-assets/releases/download/models/open_nsfw.onnx",
								   model_dir="weights",file_name="open_nsfw.onnx",progress=True)
	content_analyser = onnxruntime.InferenceSession(model_path, providers = decode_execution_providers(['cuda' if torch.cuda.is_available() else 'cpu']))
	frame = prepare_frame(frame)
	probability = content_analyser.run(None,
	{
		'input:0': frame
	})[0][0][1]
	return probability > MAX_PROBABILITY

def prepare_frame(frame) -> Any:
	frame = cv2.resize(frame, (224, 224)).astype(numpy.float32)
	frame -= numpy.array([ 104, 117, 123 ]).astype(numpy.float32)
	frame = numpy.expand_dims(frame, axis = 0)
	return frame


def encode_execution_providers(execution_providers : List[str]) -> List[str]:
	return [ execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers ]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
	available_execution_providers = onnxruntime.get_available_providers()
	encoded_execution_providers = encode_execution_providers(available_execution_providers)
	return [ execution_provider for execution_provider, encoded_execution_provider in zip(available_execution_providers, encoded_execution_providers) if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers) ]

