import cv2
import torch
from torchaudio.io import StreamReader, StreamWriter


class AVStreamer:
    '''
    Fast reading of audio and video, chunk by chunk, using torchaudio. Useful for torch style datasets where __getitem__
    is called multiple times. A chunk can be retrieved instead of reading the whole video everytime
    '''

    @staticmethod
    def stream_generator(video_file: str,
                         audio_sample_rate: int =16000,
                         chunk_size_secs: int = 1,) -> StreamReader:
        '''
        Returns a generator objects that yields chunks of video and audio of given chunk size.
        Audio is resampled to the given sample rate.

        For e.g
        a chunk size of 1 second of 25 fps video will yield 25 frames of video and 16000 samples of audio.

        Usage:
        streamer = AVStreamer.stream_generator(video_file, chunk_size_secs=1)
        stream_gen = streamer.stream()
        video_chunk, audio_chunk = next(stream_gen)
        '''

        streamer = StreamReader(video_file)
        cap = cv2.VideoCapture(video_file)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        streamer.add_basic_video_stream(frames_per_chunk=int(chunk_size_secs*fps),
                                        frame_rate=fps,
                                        format='rgb24', )
        streamer.add_basic_audio_stream(frames_per_chunk=int(chunk_size_secs*audio_sample_rate),
                                        sample_rate=16000, )
        return streamer


    @staticmethod #TODO: Fix me
    def chunks_to_output(output_path: str,
                         video_chunks: torch.Tensor,
                         audio_chunks: torch.Tensor,
                         video_fps: int = 25,
                         audio_sample_rate: int = 16000,
                         ) -> None:
        '''
        Writes chunks to a video
        :param video_chunks: BCHW tensor
        :param audio_chunks: BCN  tensor
        '''
        writer = StreamWriter(output_path)
        H,W      = video_chunks[0].shape[2:4]
        writer.add_video_stream(frame_rate=video_fps, width=W, height=H, format='rgb24')
        writer.add_audio_stream(sample_rate=audio_sample_rate, num_channels=audio_chunks.shape[1])

        # Write video frames to file
        writer.open()
        for video_chunk, audio_chunk in zip(video_chunks, audio_chunks):
            if any(x is None for x in (video_chunk, audio_chunk)):
                break
            writer.write_video_chunk(0, video_chunk)
            writer.write_audio_chunk(1, audio_chunk)
        writer.close()
