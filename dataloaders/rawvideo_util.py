import torch as th
import numpy as np
from PIL import Image
# pytorch=1.7.1
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
# pip install opencv-python
import cv2


'''
# The extractor class
Responsiblities:
1. Reading video
2. Finding its FPS and number of frames
3. Sampling frames
4. Converting BGR → RGB
5. Resizing/cropping
6. Converting pixels to PyTorch tensors
7. Normalizing for CLIP
'''
class RawVideoExtractorCV2():
    def __init__(self, centercrop=False, size=224, framerate=-1, ):       
        #The centercrop parameter is slightly misleading here:
        #it is stored but not actually used to choose the transform.
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate
        self.transform = self._transform(self.size)

    def _transform(self, n_px):
        return Compose([          
            # The image is resized so its shorter side becomes 224 while preserving the aspect ratio.
            Resize(n_px, interpolation=Image.BICUBIC),
            # Becomes: 224 × 224 by cropping the center.
            CenterCrop(n_px),
            # This handles potentially grayscale or unusual image formats.
            lambda image: image.convert("RGB"),           
            # PIL image:
            # [H, W, C], uint8, 0 → 255
            # becomes approximately:
            # [C, H, W], float, 0.0 → 1.0
            ToTensor(),            
            # These are the CLIP normalization statistics.
            # This means the video preprocessing is designed for a model expecting CLIP-style image inputs.
            # A major assumption is:
            # The downstream vision encoder expects these exact normalization statistics.
            # If you feed the result into a different model, these values may be inappropriate.
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    # 1. sample selected frames 2.preprocess every frame, 3.stack them, 4. PyTorch tensor
    def video_to_tensor(self, video_file, preprocess, sample_fp=0, start_time=None, end_time=None):
        
        # start_time and end_time are used for temporal cropping
        # time is expressed in whole seconds; you cannot specify only start_time
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                   and start_time > -1 and end_time > start_time
            
        # this assertion would fail unless something overrides self.framerate on get_video_data call
        assert sample_fp > -1

        # Samples a frame sample_fp X frames.
        cap = cv2.VideoCapture(video_file)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # No video has duratino of zero!, default start and end time also set 
        total_duration = (frameCount + fps - 1) // fps
        start_sec, end_sec = 0, total_duration

        # Attempts to seek the frame location of start_time 
        if start_time is not None:
            start_sec, end_sec = start_time, end_time if end_time <= total_duration else total_duration
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))

        
        # sample_fp is number of frames to sample in 1 second of video
        # fps is number of frames present in one second of video
        # Define sampling interval
        interval = 1
        if sample_fp > 0:
            interval = fps // sample_fp
        else:
            sample_fp = fps
        if interval == 0: interval = 1

        # the sampling is approximately uniform but not mathematically perfect.
        # sampling happens for each second
        # inds are frame offsets inside one second (the begining of second is offset 0)
        # also applies the transforms named with preprocess
        inds = [ind for ind in np.arange(0, fps, interval)]
        assert len(inds) >= sample_fp
        inds = inds[:sample_fp]

        ret = True
        images, included = [], []

        
        ## for every second:
        ##    extract N frames
        for sec in np.arange(start_sec, end_sec + 1):
            if not ret: break
            sec_base = int(sec * fps)
            for ind in inds:
                cap.set(cv2.CAP_PROP_POS_FRAMES, sec_base + ind)
                ret, frame = cap.read()
                if not ret: break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))

        cap.release()

        if len(images) > 0:
            video_data = th.tensor(np.stack(images))
        else:
            video_data = th.zeros(1)
        return {'video': video_data}

    
    def get_video_data(self, video_path, start_time=None, end_time=None):
        image_input = self.video_to_tensor(video_path, self.transform, sample_fp=self.framerate, start_time=start_time, end_time=end_time)
        return image_input
    
    # (L × T × 3 × H × W)
    # L = number of temporal units / frames
    # T = number of frames inside each unit
    # always T=1 because this extractor treats each sampled frame as an independent one-frame clip.
    def process_raw_data(self, raw_video_data):
        tensor_size = raw_video_data.size()
        tensor = raw_video_data.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
        return tensor

    def process_frame_order(self, raw_video_data, frame_order=0):
        # 0: ordinary order; 1: reverse order; 2: random order.
        if frame_order == 0:
            pass
        elif frame_order == 1:
            reverse_order = np.arange(raw_video_data.size(0) - 1, -1, -1)
            raw_video_data = raw_video_data[reverse_order, ...]
        elif frame_order == 2:
            random_order = np.arange(raw_video_data.size(0))
            np.random.shuffle(random_order)
            raw_video_data = raw_video_data[random_order, ...]

        return raw_video_data

# An ordinary video frame extractor based CV2
# Alias: The rest of code deosn't care about the underlying CV2 implementation.
RawVideoExtractor = RawVideoExtractorCV2