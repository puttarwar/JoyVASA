import sys
from pathlib import Path

import torch.utils.data

sys.path.append(str(Path(__file__).parents[1].resolve()))
import numpy as np
import yaml
import pickle



class LEAPFeatsDataset(torch.utils.data.Dataset):
    """
    Dataset class for LEAP Audio Video Clips.
    """

    def __init__(self,
                 root_dir: str,
                 leap_av_clips_dir: str,
                 chunk_history_secs: int = 1,  # Past audio/video to condition on
                 chunk_pred_secs: int = 1,     # Future audio: conditioning, Future video: prediction
                 pattern: str = '**/*.pkl',
                 ):

        super().__init__()

        self._leap_av_clips_dir = leap_av_clips_dir
        self._root_dir          = root_dir
        self._chunk_history_secs = chunk_history_secs
        self._chunk_pred_secs    = chunk_pred_secs
        self._chunk_length_secs  = chunk_history_secs + chunk_pred_secs

        # Get items from config
        config                   = yaml.load(open(Path(self._root_dir).parent / 'config.yaml'), Loader=yaml.FullLoader)
        self._video_fps          = config['video_fps']
        self._audio_sr           = config['audio_sr']
        self._audio_feat_sr      = 50  # Each audio feature is 20ms, so 50 features per second (Wav2Vec2)
        self._items              = sorted(list(Path(self._root_dir).glob(pattern)))
        self._rng                = np.random.RandomState(0)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        is_correct = False
        item_rng  = np.random.RandomState(idx)
        while is_correct is False:
            file_path = item_rng.choice(self._items)
            data_path = str(Path(self._root_dir) / file_path)
            clip_path = str(Path(self._leap_av_clips_dir) / file_path)
            data      = pickle.load(open(data_path, 'rb'))
            zs        = data['zs']
            audio_feats = data['audio_feats'].squeeze()

            clip_duration = min(len(zs)/self._video_fps, len(audio_feats)/self._audio_feat_sr)
            start_time    = self._rng.uniform(0, clip_duration - self._chunk_length_secs)
            is_correct    = start_time > 0

        z_src  = zs[int(start_time*self._video_fps):int((start_time+self._chunk_history_secs)*self._video_fps)]
        z_tgt  = zs[int((start_time+self._chunk_history_secs)*self._video_fps):int((start_time+self._chunk_length_secs)*self._video_fps)]    
        past_audio_feats = audio_feats[int(start_time*self._audio_feat_sr):int((start_time+self._chunk_history_secs)*self._audio_feat_sr)]
        curr_audio_feats = audio_feats[int((start_time+self._chunk_history_secs)*self._audio_feat_sr):
                                       int((start_time+self._chunk_length_secs)*self._audio_feat_sr)]


        return {'z_src': z_src,
                'z_tgt': z_tgt,
                'past_audio_feats': past_audio_feats,
                'curr_audio_feats': curr_audio_feats,
                'data_path': data_path,
                'clip_path': clip_path,}


if __name__ == '__main__':
    # dataset = LEAPAudioVideoClips(root_dir = '/hdd/LEAPclips-av/',
    #                               resolution=256,
    #                               pattern='/**/*.mp4',
    #                               augmentation={'resolution': 256},
    #                               chunk_history_secs=1,
    #                               chunk_pred_secs=1,
    #                               )

    dataset = LEAPFeatsDataset(root_dir = '/hdd/model-zoo/test_model_v2.4.1/superlight/250415-1319-lsm_audio2video_dataset/dataset/',
                               leap_av_clips_dir='/hdd/LEAPclips-av/',
                               pattern='**/*.pkl',
                               chunk_history_secs=1,
                               chunk_pred_secs=1,
                               )
    dataset._rng = np.random.RandomState(0)
    for i in range(2):
        sample = dataset[i]
        print(sample['data_path'])
        print(sample['z_src'].shape)
        print(sample['z_tgt'].shape)
        print(sample['audio_feats'].shape)
        print(f'{"-"*80}')
