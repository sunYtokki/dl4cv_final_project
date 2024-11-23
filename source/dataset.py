import os
import torch
import librosa
import numpy as np

class DnR_Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, sample_rate=44100, transform=None, chunk_size=132300, offset=None):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.transform = transform
        self.chunk_size = chunk_size
        self.offset = offset
        self.folders = [
            os.path.join(root_dir, folder)
            for folder in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, folder))
        ]

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder = self.folders[idx]

        # Check for required files
        files = {
            "mix": os.path.join(folder, "mix.wav"),
            "speech": os.path.join(folder, "speech.wav"),
            "music": os.path.join(folder, "music.wav"),
            "sfx": os.path.join(folder, "sfx.wav"),
        }
        for key, path in files.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {key} file: {path}")

        # Load and preprocess audio
        # mix = self._load_chunk(files["mix"])
        # mix = self._audio_to_tensor(mix).unsqueeze(0).to(self.device)  # Shape: [1, channels, samples]

        mix = self._audio_to_tensor(self._load_chunk(files["mix"]))
        speech = self._audio_to_tensor(self._load_chunk(files["speech"]))
        music = self._audio_to_tensor(self._load_chunk(files["music"]))
        sfx = self._audio_to_tensor(self._load_chunk(files["sfx"]))

        # Align lengths
        # target_length = mix.shape[-1]
        # speech = self._pad_or_trim(speech, target_length)
        # music = self._pad_or_trim(music, target_length)
        # sfx = self._pad_or_trim(sfx, target_length)

        # Stack targets
        targets = torch.stack([speech, music, sfx], dim=0)  # Shape: [num_stems, channels, samples]

        # Apply transforms
        if self.transform:
            mix = self.transform(mix)
            targets = self.transform(targets)

        return mix, targets

    def _load_audio_from_path(self, path):
        # Load the audio using librosa
        audio, sr = librosa.load(path, mono=False, sr=self.sample_rate)
        return self._mono_to_stereo(audio)

    def _mono_to_stereo(self, audio):
        # Convert mono audio to stereo if needed
        if len(audio.shape) == 1:
            audio = np.stack([audio, audio], axis=0)  # Duplicate channel for stereo
        return audio

    def _audio_to_tensor(self, audio):
        # Convert numpy array to PyTorch tensor
        return torch.from_numpy(audio).float()

    def _pad_or_trim(self, audio, target_length):
        # Pad or trim audio to the target length
        if audio.shape[-1] < target_length:
            pad_width = target_length - audio.shape[-1]
            audio = torch.nn.functional.pad(audio, (0, pad_width))
        elif audio.shape[-1] > target_length:
            audio = audio[..., :target_length]
        return audio

    def _load_chunk(self, path):
        # Load the audio with librosa
        audio, sr = librosa.load(path, sr=self.sample_rate, mono=False)

        # Ensure the audio is stereo
        audio = self._mono_to_stereo(audio)

        # Determine the total length of the audio
        total_samples = audio.shape[1]

        # Handle chunking
        if self.chunk_size <= total_samples:
            # Determine the offset
            if self.offset is None:
                offset = np.random.randint(0, total_samples - self.chunk_size + 1)
            else:
                offset = self.offset
            audio = audio[:, offset:offset + self.chunk_size]
        else:
            # If chunk size exceeds audio length, pad the audio
            pad_width = self.chunk_size - total_samples
            audio = np.pad(audio, ((0, 0), (0, pad_width)), mode="constant")

        return audio