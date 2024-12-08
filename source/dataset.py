# forked from https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/a45d9bc7889184efecec3f79ec4d06143afa7d4a/dataset.py

import os
import random
import numpy as np
import torch
import soundfile as sf
import pickle
import time
import itertools
import multiprocessing
from tqdm.auto import tqdm
from glob import glob
import warnings
warnings.filterwarnings("ignore")

def _mono_to_stereo(audio):
    """
    Convert mono audio to stereo by duplicating the channel if needed.
    """
    if len(audio.shape) == 1:  # Check if the audio is mono
        audio = np.stack([audio, audio], axis=1)  # Duplicate the channel to make it stereo
    return audio

def load_chunk(path, length, chunk_size, offset=None):
    if chunk_size <= length:
        if offset is None:
            offset = np.random.randint(length - chunk_size + 1)
        x = sf.read(path, dtype='float32', start=offset, frames=chunk_size)[0]
    else:
        x = sf.read(path, dtype='float32')[0]
        pad = np.zeros([chunk_size - length, 2])
        x = np.concatenate([x, pad])
    # Mono fix
    x = _mono_to_stereo(x)
    return x.T

def get_track_set_length(params):
    path, instruments, file_types = params
    # Check lengths of all instruments (it can be different in some cases)
    lengths_arr = []
    for instr in instruments:
        length = -1
        for extension in file_types:
            path_to_audio_file = path + '/{}.{}'.format(instr, extension)
            if os.path.isfile(path_to_audio_file):
                length = len(sf.read(path_to_audio_file)[0])
                break
        if length == -1:
            print('Cant find file "{}" in folder {}'.format(instr, path))
            continue
        lengths_arr.append(length)
    lengths_arr = np.array(lengths_arr)
    if lengths_arr.min() != lengths_arr.max():
        print('Warning: lengths of stems are different for path: {}. ({} != {})'.format(
            path,
            lengths_arr.min(),
            lengths_arr.max())
        )
    # We use minimum to allow overflow for soundfile read in non-equal length cases
    return path, lengths_arr.min()


# For multiprocessing
def get_track_length(params):
    path = params
    length = len(sf.read(path)[0])
    return (path, length)


class MSSDataset(torch.utils.data.Dataset):
    def __init__(self, instruments, data_path, sr=44100, segment=6, metadata_path="metadata.pkl", batch_size=None, num_steps = 100, verbose=True):
        self.verbose = verbose
        self.data_path = data_path
        self.instruments = instruments 
        self.batch_size = batch_size
        self.file_types = ['wav']
        self.metadata_path = metadata_path
        self.num_steps = num_steps
        self.chunk_size = sr*segment
        self.min_mean_abs = 0.001

        metadata = self.get_metadata()

        if len(metadata) > 0:
            if self.verbose:
                print('Found tracks in dataset: {}'.format(len(metadata)))
        else:
            print('No tracks found for training. Check paths you provided!')
            exit()

        self.metadata = metadata

    def __len__(self):
        return self.num_steps * self.batch_size

    def read_from_metadata_cache(self, track_paths, instr=None):
        metadata = []
        if os.path.isfile(self.metadata_path):
            if self.verbose:
                print('Found metadata cache file: {}'.format(self.metadata_path))
            old_metadata = pickle.load(open(self.metadata_path, 'rb'))
        else:
            return track_paths, metadata

        if instr:
            old_metadata = old_metadata[instr]

        # We will not re-read tracks existed in old metadata file
        track_paths_set = set(track_paths)
        for old_path, file_size in old_metadata:
            if old_path in track_paths_set:
                metadata.append([old_path, file_size])
                track_paths_set.remove(old_path)
        track_paths = list(track_paths_set)
        if len(metadata) > 0:
            print('Old metadata was used for {} tracks.'.format(len(metadata)))
        return track_paths, metadata


    def get_metadata(self):
        read_metadata_procs = multiprocessing.cpu_count()

        track_paths = []
        if type(self.data_path) == list:
            for tp in self.data_path:
                tracks_for_folder = sorted(glob(tp + '/*'))
                if len(tracks_for_folder) == 0:
                    print('Warning: no tracks found in folder \'{}\'. Please check it!'.format(tp))
                track_paths += tracks_for_folder
        else:
            track_paths += sorted(glob(self.data_path + '/*'))

        track_paths = [path for path in track_paths if os.path.basename(path)[0] != '.' and os.path.isdir(path)]
        track_paths, metadata = self.read_from_metadata_cache(track_paths, None)

        if read_metadata_procs <= 1:
            for path in tqdm(track_paths):
                track_path, track_length = get_track_set_length((path, self.instruments, self.file_types))
                metadata.append((track_path, track_length))
        else:
            p = multiprocessing.Pool(processes=read_metadata_procs)
            with tqdm(total=len(track_paths)) as pbar:
                track_iter = p.imap(
                    get_track_set_length,
                    zip(track_paths, itertools.repeat(self.instruments), itertools.repeat(self.file_types))
                )
                for track_path, track_length in track_iter:
                    metadata.append((track_path, track_length))
                    pbar.update()
            p.close()

        # Save metadata
        pickle.dump(metadata, open(self.metadata_path, 'wb'))
        return metadata

    def load_source(self, metadata, instr):
        while True:
            track_path, track_length = random.choice(metadata)
            for extension in self.file_types:
                path_to_audio_file = track_path + '/{}.{}'.format(instr, extension)
                if os.path.isfile(path_to_audio_file):
                    try:
                        source = load_chunk(path_to_audio_file, track_length, self.chunk_size)
                    except Exception as e:
                        # Sometimes error during FLAC reading, catch it and use zero stem
                        print('Error: {} Path: {}'.format(e, path_to_audio_file))
                        source = np.zeros((2, self.chunk_size), dtype=np.float32)
                    break
            if np.abs(source).mean() >= self.min_mean_abs:  # remove quiet chunks
                break
        return torch.tensor(source, dtype=torch.float32)

    def load_random_mix(self):
        res = []
        for instr in self.instruments:
            s1 = self.load_source(self.metadata, instr)
            res.append(s1)
        res = torch.stack(res)
        return res

    def __getitem__(self, index):
        res = self.load_random_mix()
        mix = res.sum(0)
        return res, mix
