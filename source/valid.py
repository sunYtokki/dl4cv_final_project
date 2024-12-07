import torch
from tqdm import tqdm
import numpy as np
import time
import librosa
import glob
import soundfile as sf
import torch.nn as nn
import os

def sdr(references, estimates):
    # references and estimates: [N, time, channels]
    delta = 1e-7  # avoid numerical errors
    num = np.sum(np.square(references), axis=(1, 2))
    den = np.sum(np.square(references - estimates), axis=(1, 2))
    num += delta
    den += delta
    return 10 * np.log10(num / den)

def get_predicted_audio_stems(model, sr, mix, device):
    #TODO: parametrize configuration
    segment = 6
    overlap = 4
    batch_size = 8

    S = len(model.sources)
    C = sr * segment 
    N = overlap
    step = C // N

    with torch.cuda.amp.autocast(enabled=True):
        with torch.inference_mode():

            if mix.shape[0] == 1:  # Check if the input is mono
                mix = mix.expand(2, -1)  # Duplicate channel to create stereo

            req_shape = (S, ) + tuple(mix.shape)
            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)
            i = 0
            batch_data = []
            batch_locations = []
            progress_bar = tqdm(total=mix.shape[1], desc="Processing audio chunks", leave=False)

            while i < mix.shape[1]:
                # print(i, i + C, mix.shape[1])
                part = mix[:, i:i + C].to(device)
                length = part.shape[-1]
                if length < C:
                    part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)
                batch_data.append(part)
                batch_locations.append((i, length))
                i += step


                if len(batch_data) >= batch_size or (i >= mix.shape[1]):
                    arr = torch.stack(batch_data, dim=0)

                    x = model(arr)
                    for j in range(len(batch_locations)):
                        start, l = batch_locations[j]
                        result[..., start:start+l] += x[j][..., :l].cpu()
                        counter[..., start:start+l] += 1.
                    batch_data = []
                    batch_locations = []

                if progress_bar:
                    progress_bar.update(step)

            if progress_bar:
                progress_bar.close()

            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

    if S > 1:
        return {k: v for k, v in zip(model.sources, estimated_sources)}
    else:
        return estimated_sources
    

def read_audio_chunk(file_path, chunk_size, target_sr, start_sample=0):
    """
    Reads a single chunk of audio data from the file.

    Args:
        file_path (str): Path to the audio file.
        chunk_size (int): Number of samples to read.
        target_sr (int): Target sample rate for resampling.
        start_sample (int): Starting sample index for the chunk.

    Returns:
        chunk (numpy.ndarray): Audio chunk [channels, samples].
        sr (int): Sample rate.
    """
    with sf.SoundFile(file_path) as audio_file:
        sr = audio_file.samplerate
        audio_file.seek(start_sample)

        # Read a chunk of data
        chunk = audio_file.read(chunk_size, dtype='float32', always_2d=True).T  # Shape: (channels, samples)

        # Resample if needed
        if sr != target_sr:
            chunk = librosa.resample(chunk, orig_sr=sr, target_sr=target_sr, res_type='kaiser_best')

        return chunk, target_sr


def valid(model, valid_path, extension='wav', target_sr=44100, device='cpu', verbose=False):
    """
    Validation function: loads sound files directly, ensures correct dimensions, and calculates SDR.
    """
    start_time = time.time()
    model.eval().to(device)

    # Use the instrument order from the model to ensure consistency
    instruments = model.sources
    sdr_values = {instr: [] for instr in instruments}

    valid_paths = []
    for path in valid_path:
        valid_paths += sorted(glob.glob(os.path.join(path, '*', f'mix.{extension}')))

    print(f"Found {len(valid_paths)} mixtures for validation.")

    chunk_size = 6 * target_sr

    with torch.no_grad():
        for mix_path in tqdm(valid_paths, desc="Validation", disable=not verbose):
            # Load mixture
            # mix, sr = read_audio_chunk(mix_path, chunk_size, target_sr)

            mix, sr = sf.read(mix_path)  # shape: (samples,) or (samples, channels)
            # Ensure sample rate and channel matches for mix
            if sr != target_sr:
                mix = librosa.resample(mix, orig_sr=sr, target_sr=target_sr, res_type='kaiser_best')
                sr = target_sr
            if len(mix.shape) == 1:
                mix = np.expand_dims(mix, axis=-1)

            mix = mix.T #(ch, waveform)

            mix = torch.tensor(mix, dtype=torch.float32)
            waveforms = get_predicted_audio_stems(model, target_sr, mix, device)

            folder = os.path.dirname(mix_path)
            for instr in instruments:
                gt_path = os.path.join(folder, f"{instr}.{extension}")
                if not os.path.exists(gt_path):
                    if verbose:
                        print(f"Warning: Ground truth file missing for {instr} in {folder}. Skipping...")
                    continue

                # Load targets
                # target_source, sr = read_audio_chunk(mix_path, chunk_size, target_sr)

                target_source, sr_gt = sf.read(gt_path)
                # Ensure sample rate and channel matches for target 
                if sr_gt != sr:
                    target_source = librosa.resample(target_source, orig_sr=sr_gt, target_sr=sr, res_type='kaiser_best')
                if len(target_source.shape) == 1:
                    target_source = np.expand_dims(target_source, axis=-1)

                estimated_source = waveforms[instr]  # (2, samples)

                # compute SDR for each stem:
                # current shape: (2, samples)
                # need (1, samples, 2)
                references = np.expand_dims(target_source.T, axis=0)   # (1, samples, 2)
                estimates = np.expand_dims(estimated_source, axis=0) # (1, samples, 2)
                sdr_val = sdr(references, estimates)[0]
                sdr_values[instr].append(sdr_val)

    # Compute average SDR
    metrics_avg = {}
    for instrument in sdr_values:
        vals = sdr_values[instrument]
        if len(vals) > 0:
            mean_val = np.mean(vals)
            std_val = np.std(vals)
        else:
            mean_val, std_val = 0.0, 0.0
        print(f"Average SDR for {instrument}: {mean_val:.4f} dB (Std: {std_val:.4f} dB)")
        metrics_avg[f'sdr_{instrument}'] = mean_val

    all_sdr_vals = [val for instr_vals in sdr_values.values() for val in instr_vals]
    overall_sdr = np.mean(all_sdr_vals) if all_sdr_vals else 0.0
    metrics_avg['sdr'] = overall_sdr
    print(f"Average SDR over all tracks: {overall_sdr:.4f} dB")

    print("Elapsed time: {:.2f} sec".format(time.time() - start_time))
    return metrics_avg