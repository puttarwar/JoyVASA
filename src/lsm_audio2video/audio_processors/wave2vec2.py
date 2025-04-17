import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class Wav2Vec2(nn.Module):
    ''' Wav2Vec2 feature extractor for audio processing.'''
    def __init__(self,
                 sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate

        # Define models
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.device = torch.device('cuda:0')

    def forward(self, audio: torch.Tensor| np.ndarray):
        '''
        Forward pass through the Wav2Vec2 model to extract features.
        :param audio: [batch_size, seq_len] tensor or numpy array
        :return: features: [batch_size, seq_len_chunked, feature_dim] tensor

        seq_len_chunked = seq_len // chunk_size
        '''
        # Preprocess the audio
        try:
            inputs = self.processor(audio, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
        except Exception as e:
            inputs = self.processor(audio.cpu().numpy(), sampling_rate=self.sample_rate, return_tensors="pt", padding=True)

        # Forward pass through the model
        with torch.no_grad():
            if inputs.input_values.dim() == 3:
                inputs.input_values = inputs.input_values.squeeze(0)
            out = self.model(inputs.input_values.to(self.device), output_hidden_states=True)

        # Extract last layer features
        feats = out.hidden_states[-1]

        # Get Transcrption for sanity checking (Optional)
        # predicted_ids = torch.argmax(out.logits, dim=-1)
        # transcription = self.processor.batch_decode(predicted_ids)

        return feats