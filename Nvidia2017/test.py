import librosa
from scipy.io import wavfile


def process_audio(ds_path, audio, sample_rate):
    config = {}
    config['deepspeech_graph_fname'] = ds_path
    config['audio_feature_type'] = 'deepspeech'
    config['num_audio_features'] = 29

    config['audio_window_size'] = 16
    config['audio_window_stride'] = 1

    tmp_audio = {'subj': {'seq': {'audio': audio, 'sample_rate': sample_rate}}}
    audio_handler = AudioHandler(config)
    return audio_handler.process(tmp_audio)['subj']['seq']['audio']


if __name__ == '__main__':
    # 指定要读取的 WAV 文件路径
    wav_file_path = "../wave/angry.wav"

    # 读取 WAV 文件中的音频数据和采样率
    audio_data, sample_rate = wavfile.read(wav_file_path)

    # 打印音频数据和采样率
    print("Audio data shape:", audio_data)
    print("Sample rate:", sample_rate)
