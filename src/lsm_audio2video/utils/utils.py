from functools import reduce
from pathlib import Path
from collections.abc import Sequence

import cv2
import pandas as pd
import torch

from src.lsm_audio2video.utils.bbox import BBox
from src.lsm_audio2video.utils.torch_av_streamer import AVStreamer

AUDIO_SAMPLE_RATE = 16000
VIDEO_FPS         = 25


def get_leap_clip_data(video_path, device, resolution=256):
    streamer = AVStreamer.stream_generator(video_path, chunk_size_secs=1,
                                           audio_sample_rate=AUDIO_SAMPLE_RATE,)

    # Get fps
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    # Fetch audio video
    video = []
    audio = []
    for video_chunk, audio_chunk in streamer.stream():
        if video_chunk is None or audio_chunk is None:
            break
        video.append(video_chunk)
        audio.append(audio_chunk)

    video = torch.cat(video, dim=0).to(device)
    audio = torch.cat(audio, dim=0).to(device)

    # Crop audio to have same size as video
    audio = audio[:int(len(video) * 16000 / fps)]

    # Resample video at constant fps
    resample_idxs = torch.linspace(0, len(video) - 1, int(len(video) * VIDEO_FPS / fps)).long().to(device)
    video = video[resample_idxs]

    # Crop according to union bbox
    anno = pd.read_csv(Path(video_path).with_suffix('.csv'))
    bboxes = [BBox(*bbox) for bbox in anno[['left', 'top', 'right', 'bottom']].values]
    union_bbox = reduce(lambda x, y: x | y, bboxes)
    video = video[:, :, union_bbox.y1:union_bbox.y2, union_bbox.x1:union_bbox.x2]

    # Process video
    video = video/127.5 - 1
    video = torch.nn.functional.interpolate(video,size=(resolution, resolution), mode='bilinear', align_corners=False)

    return video, audio


def batch_broadcast(enc, b):
    enc_clone = enc.copy()
    for k,v in enc_clone.items():
        if isinstance(v, torch.Tensor):
            enc_clone[k] = v.expand(b, *v.shape[1:])
        if isinstance(v,Sequence):
            enc_clone[k] = [vv.expand(b, *vv.shape[1:]) for vv in v]
    return enc_clone
