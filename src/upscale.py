import os
import torch
import glob
import cv2
from tqdm import tqdm
from typing import Any
import subprocess, platform
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.archs.rrdbnet_arch import RRDBNet
from src.third_part.codeformer.codeformer_arch import CodeFormer
from basicsr.utils.download_util import load_file_from_url
from src.third_part.facelib.utils.face_restoration_helper import FaceRestoreHelper
from realesrgan import RealESRGANer
from src.log_helper import HandleLog
import threading
from joblib import Parallel,delayed
from src.third_part.codeformer.video_util import VideoWriter,VideoReader

logger = HandleLog()
num_lock = threading.Lock()

class Upscale:
    def __init__(self,fidelity_weight=0.9) -> None:
        self.pretrain_model_url = {
            'restoration': 'https://mirror.ghproxy.com/https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_half = True 
        self.bg_tile = 400
        self.w = fidelity_weight
        self.bg_upsampler = 'realesrgan'
        self.face_upsample = True
        self.has_aligned = False
        self.detection_model = "retinaface_resnet50"
        self.upscale = 2
        self.only_center_face = False
        self.draw_box = False
        self.suffix = None


    def __call__(self,input_path:str,output_path:str,audio,*args: Any, **kwds: Any) -> Any:
        
        input_video = False
        if input_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single img path
            input_img_list = [input_path]
            result_root = f'results/test_img_{self.w}'
        elif input_path.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')): # input video path
            input_img_list = []
            vidreader = VideoReader(input_path)
            image = vidreader.get_frame()
            while image is not None:
                input_img_list.append(image)
                image = vidreader.get_frame()
            # audio = vidreader.get_audio()
            fps = vidreader.get_fps()    
            video_name = os.path.basename(input_path)[:-4]
            result_root = f'results/{video_name}_{self.w}'
            input_video = True
            vidreader.close()
        else: # input img folder
            if input_path.endswith('/'):  # solve when path ends with /
                input_path = input_path[:-1]
            # scan all the jpg and png images
            input_img_list = sorted(glob.glob(os.path.join(input_path, '*.[jpJP][pnPN]*[gG]')))
            result_root = f'results/{os.path.basename(input_path)}_{self.w}'
        
        if not output_path is None: # set output path
            result_root = output_path

        test_img_num = len(input_img_list)
        if test_img_num == 0:
            raise FileNotFoundError('No input image/video is found...\n' 
                '\tNote that --input_path for video should end with .mp4|.mov|.avi')

        # ------------------ set up background upsampler ------------------
        if self.bg_upsampler == 'realesrgan':
            bg_upsampler = self.set_realesrgan()
        else:
            bg_upsampler = None
        
        # ------------------ set up face upsampler ------------------
        if self.face_upsample:
            if bg_upsampler is not None:
                face_upsampler = bg_upsampler
            else:
                face_upsampler = self.set_realesrgan()
        else:
            face_upsampler = None
        
         # ------------------ set up CodeFormer restorer -------------------
        net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                                                connect_list=['32', '64', '128', '256']).to(self.device)
        # ckpt_path = 'weights/CodeFormer/codeformer.pth'
        ckpt_path = load_file_from_url(url=self.pretrain_model_url['restoration'], 
                                        model_dir='weights/CodeFormer', progress=True, file_name=None)
        checkpoint = torch.load(ckpt_path)['params_ema']
        net.load_state_dict(checkpoint)
        net.eval()

        # ------------------ set up FaceRestoreHelper -------------------
        # large det_model: 'YOLOv5l', 'retinaface_resnet50'
        # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
        if not self.has_aligned: 
            logger.info(f'Face detection model: {self.detection_model}')
        if bg_upsampler is not None: 
            logger.info(f'Background upsampling: True, Face upsampling: {self.face_upsample}')
        else:
            logger.info(f'Background upsampling: False, Face upsampling: {self.face_upsample}')

        # -------------------- start to processing ---------------------
        logger.info("multi thread processing ")
        '''
        with ThreadPoolExecutor(max_workers=20) as executor:
            for i, img_path in enumerate(input_img_list):
                executor.submit(self.enhance_face,img_path,i,video_name,test_img_num,
                                bg_upsampler,result_root,input_video,net,face_upsampler)
        '''
        Parallel(n_jobs=-1)(delayed(self.enhance_face)(img_path,i,video_name,test_img_num,\
                                                       bg_upsampler,result_root,input_video,\
                                                        net,face_upsampler) for i,img_path in enumerate(input_img_list))

        # save enhanced video
        if input_video:
            logger.info('Video Saving...')
            # load images
            video_frames = []
            img_list = sorted(glob.glob(os.path.join(result_root, 'final_results', '*.[jp][pn]g')))
            for img_path in img_list:
                img = cv2.imread(img_path)
                video_frames.append(img)
            # write images to video
            height, width = video_frames[0].shape[:2]
            if self.suffix is not None:
                video_name = f'{video_name}_{self.suffix}.png'
            save_restore_path = os.path.join(result_root, f'{video_name}.avi')
            vidwriter = cv2.VideoWriter(save_restore_path,cv2.VideoWriter_fourcc(*'DIVX'),fps, (width, height))
    
            for f in tqdm(video_frames,desc="Combining png to avi...",total=len(video_frames)):
                vidwriter.write(f)
            
            vidwriter.release()
        
        out_file = os.path.join(result_root, f'{video_name}.mp4')
        command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio, save_restore_path, out_file)
        subprocess.call(command, shell=platform.system() != 'Windows')

        logger.info(f'\nAll results are saved in {result_root}')

    def enhance_face(self,img_path,i,video_name,test_img_num,bg_upsampler,result_root,input_video,net,face_upsampler):
        # clean all the intermediate results to process the next image
        face_helper = FaceRestoreHelper(
            self.upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model = self.detection_model,
            save_ext='png',
            use_parse=True,
            device=self.device)
        with num_lock:
            if isinstance(img_path, str):
                img_name = os.path.basename(img_path)
                basename, ext = os.path.splitext(img_name)
                logger.info(f'[{i+1}/{test_img_num}] Processing: {img_name}')
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            else: # for video processing
                basename = str(i).zfill(6)
                img_name = f'{video_name}_{basename}' if input_video else basename
                logger.info(f'[{i+1}/{test_img_num}] Processing: {img_name}')
                img = img_path

        if self.has_aligned: 
            # the input faces are already cropped and aligned
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            # face_helper.is_gray = is_gray(img, threshold=10)
            if face_helper.is_gray:
                logger.info('Grayscale input: True')
            face_helper.cropped_faces = [img]
        else:
            face_helper.read_image(img)
            # get face landmarks for each face
            num_det_faces = face_helper.get_face_landmarks_5(
                only_center_face=self.only_center_face, resize=640, eye_dist_threshold=5)
            logger.info(f'\tdetect {num_det_faces} faces')
            # align and warp each face
            face_helper.align_warp_face()

        # face restoration for each cropped face
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                with torch.no_grad():
                    output = net(cropped_face_t, w=self.w, adain=True)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except Exception as error:
                logger.info(f'\tFailed inference for CodeFormer: {error}')
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

            restored_face = restored_face.astype('uint8')
            face_helper.add_restored_face(restored_face, cropped_face)

        # paste_back
        if not self.has_aligned:
            # upsample the background
            if bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = bg_upsampler.enhance(img, outscale=self.upscale)[0]
            else:
                bg_img = None
            face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            if self.face_upsample and face_upsampler is not None: 
                restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=self.draw_box, face_upsampler=face_upsampler)
            else:
                restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=self.draw_box)
        
        
        # save faces
        for idx, (cropped_face, restored_face) in enumerate(zip(face_helper.cropped_faces, face_helper.restored_faces)):
            # save cropped face
            if not self.has_aligned: 
                save_crop_path = os.path.join(result_root, 'cropped_faces', f'{basename}_{idx:02d}.png')
                imwrite(cropped_face, save_crop_path)
            # save restored face
            if self.has_aligned:
                save_face_name = f'{basename}.png'
            else:
                save_face_name = f'{basename}_{idx:02d}.png'
            if self.suffix is not None:
                save_face_name = f'{save_face_name[:-4]}_{self.suffix}.png'
            save_restore_path = os.path.join(result_root, 'restored_faces', save_face_name)
            imwrite(restored_face, save_restore_path)
        
        # save restored img
        if not self.has_aligned and restored_img is not None:
            if self.suffix is not None:
                basename = f'{basename}_{self.suffix}'
            save_restore_path = os.path.join(result_root, 'final_results', f'{basename}.png')
            imwrite(restored_img, save_restore_path)


    def set_realesrgan(self):
        if torch.cuda.is_available():
            no_half_gpu_list = ['1650', '1660'] # set False for GPUs that don't support f16
            if not True in [gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list]:
                self.use_half = True
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2
        )
        upsampler = RealESRGANer(
            scale=2,
            model_path="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
            model=model,
            tile=self.bg_tile,
            tile_pad=40,
            pre_pad=0,
            half=self.use_half
        )
        if not torch.cuda.is_available():
            logger.warning('Running on CPU now! Make sure your PyTorch version matches your CUDA.')
        return upsampler
    
