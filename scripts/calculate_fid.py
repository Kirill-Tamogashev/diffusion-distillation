from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torchvision import transforms
import torch
import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

img2tensor = transforms.ToTensor()


def read_image(path: Path):
    img = Image.open(path)
    return img2tensor(img)
    

def update_fid(batch_real, batch_synth, args, fid):
    fid.update(
        torch.stack(batch_real).to(device=args.device).to(torch.uint8), 
        real=True
    )
    fid.update(
        torch.stack(batch_synth).to(device=args.device).to(torch.uint8), 
        real=False
    )


def compute_fid():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--real_data",
        type=Path,
        required=True
    )
    parser.add_argument(
        "-s", "--synthetic_data",
        type=Path,
        required=True
    )
    parser.add_argument(
        "-f", "--fid_path",
        type=Path,
        required=True
    )
    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=32
    )
    parser.add_argument(
        "-d", "--device",
        type=int,
        default=0
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True
    )
    parser.add_argument(
        "--total_iter",
        type=int,
        default=10_000
    )
    args = parser.parse_args()
    
    fid = FID(feature=64)
    fid.cuda()
    
    batch_real = []
    batch_synth = []
    n_iter = 0
    path_iterator = zip(args.real_data.iterdir(), args.synthetic_data.iterdir())
    
    for real, synth in tqdm(path_iterator, total=args.total_iter):
        if len(batch_real) == args.batch_size:
            update_fid(batch_real, batch_synth, args, fid)
            batch_real.clear()
            batch_synth.clear()
        else:
            batch_real.append(read_image(real))
            batch_synth.append(read_image(synth))
        
        if n_iter == args.total_iter:
            if batch_real and batch_synth:
                update_fid(batch_real, batch_synth, args, fid)
            break
        n_iter += 1
            
    fid_value = fid.compute()
    message = f"Checkpoint: {args.checkpoint} | FID: {fid_value :.5f} | Sample size: {args.total_iter}\n"
    print(f"\n{message}")
    with args.fid_path.open("a") as file:
        file.write(message)


if __name__ == "__main__":
    compute_fid()
