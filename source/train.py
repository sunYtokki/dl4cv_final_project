import random
import time
from tqdm.auto import tqdm
import os
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from demucs import pretrained

from dataset import MSSDataset
from valid import valid

import warnings
warnings.filterwarnings("ignore")


def masked_loss(y_, y, q, coarse=True):
    loss = torch.nn.MSELoss(reduction='none')(y_, y).transpose(0, 1)
    if coarse:
        loss = torch.mean(loss, dim=(-1, -2))
    loss = loss.reshape(loss.shape[0], -1)
    L = loss.detach()
    quantile = torch.quantile(L, q, interpolation='linear', dim=1, keepdim=True)
    mask = L < quantile
    return (loss * mask).mean()


def manual_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_not_compatible_weights(model, weights, verbose=False):
    new_model = model.state_dict()
    old_model = torch.load(weights)
    if 'state' in old_model:
        old_model = old_model['state']
    if 'state_dict' in old_model:
        old_model = old_model['state_dict']
    for el in new_model:
        if el in old_model:
            new_model[el] = old_model[el]
    model.load_state_dict(new_model)


def train_model(
    model,
    results_path=None,
    data_path=None,
    valid_path=None,
    num_epochs=100,
    instruments = ['speech', 'music', 'sfx'],
    batch_size = 6,
    q = 0.95,
    num_workers=4,
    seed=42,
    device_ids=[0],
):
    manual_seed(seed + int(time.time()))
    torch.backends.cudnn.deterministic = False
    torch.multiprocessing.set_start_method('spawn')

    # Load model and configuration
    # model, config = get_model_from_config(model_type, config_path)
    # config = OmegaConf.load(config_path)
    print("Instruments: {}".format(instruments))

    os.makedirs(results_path, exist_ok=True)

    # batch_size = config.training.batch_size * len(device_ids)
    batch_size = batch_size * len(device_ids)
    print("Metrics for training: SDR. Metric for scheduler: SDR")

    trainset = MSSDataset(
        instruments = model.sources,
        data_path=data_path,
        batch_size=batch_size,
        metadata_path=os.path.join(results_path, f'metadata.pkl'),
    )

    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    if torch.cuda.is_available():
        if len(device_ids) <= 1:
            device = torch.device(f'cuda:{device_ids[0]}')
            model = model.to(device)
        else:
            device = torch.device(f'cuda:{device_ids[0]}')
            model = nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        device = 'cpu'
        print('CUDA is not available. Running training on CPU.')
        model = model.to(device)

    optimizer = Adam(model.parameters(), lr=9e-05)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.95)
    scaler = GradScaler()
    print(f"Patience: 2 Reduce factor: 0.95 Batch size: {batch_size}")

    best_metric = -10000
    for epoch in range(num_epochs):
        model.train()
        print(f"Epoch: {epoch+1}/{num_epochs}")
        loss_val = 0.
        total = 0

        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for i, (batch, mixes) in enumerate(pbar):
            y = batch.to(device)
            x = mixes.to(device)

            # normalize input
            mean, std = x.mean(), x.std()
            x = (x - mean) / std
            y = (y - mean) / std

            with torch.cuda.amp.autocast(enabled=True):
                y_ = model(x)
                loss = masked_loss(y_, y, q=q)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            li = loss.item()
            loss_val += li
            total += 1
            pbar.set_postfix({'loss': 100 * li, 'avg_loss': 100 * (loss_val / total)})
        
        avg_loss = loss_val / total
        print(f"Training Loss: {avg_loss:.6f}")

        # Save checkpoint
        store_path = os.path.join(results_path, f'last_model.ckpt')
        torch.save(model.state_dict(), store_path)

        # Validation
        # args = SimpleNamespace(valid_path=valid_path)
        metrics_avg = valid(model=model, valid_path=valid_path, device=device, verbose=False)
        metric_avg = metrics_avg.get('sdr', 0.0)

        print(f"Validation SDR: {metric_avg:.4f}")
        if metric_avg > best_metric:
            best_metric = metric_avg
            best_path = os.path.join(results_path, f"best_model_epoch_{epoch+1}_{metric_avg:.4f}.ckpt")
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved: {best_path}")
        scheduler.step(metric_avg)

    print(f"Training complete. Best SDR: {best_metric:.4f}")


if __name__ == "__main__":

    htdemucs = pretrained.get_model('htdemucs') # load pretrained htdemucs

    # modify network to have 3 stems output
    model = htdemucs.models[0]
    model.sources = ['speech', 'music', 'sfx']
    model.decoder[-1].conv_tr = torch.nn.ConvTranspose2d(
    in_channels=48,  
    out_channels=12,  # 3 stems * input channels (2 for stereo)
    kernel_size=(8, 1),
    stride=(4, 1)
    )
    model.tdecoder[-1].conv_tr = torch.nn.ConvTranspose1d(
        in_channels=48,  
        out_channels=6,
        kernel_size=8,
        stride=4
    )

    train_model(
        model=model,
        results_path='/Users/yt/coding/DL4CV/Final/Cinematic_sound_demixer/output/result/test',
        data_path=['/Users/yt/coding/DL4CV/Final/Cinematic_sound_demixer/DnR/dnr_small/tr'],
        valid_path=['/Users/yt/coding/DL4CV/Final/Cinematic_sound_demixer/DnR/dnr_small/cv'],
        num_epochs=10
    )