import os
import argparse
from tqdm import tqdm
from huggingface_hub import snapshot_download


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--path", type=str, default="/zoo")
    parser.add_argument("--token", type=str, default=None)

    args = parser.parse_args()

    model_name = args.model
    local_dir = "{}/{}".format(args.path, args.model.split("/")[-1])
    if not os.path.exists(local_dir):
        try:
            snapshot_download(
                model_name,
                token=args.token,
                local_dir=local_dir,
                ignore_patterns=["*.bin", "*.pth", ".huggingface/*", "consolidated*"],
                tqdm_class=tqdm
            )
        except FileNotFoundError:
            snapshot_download(
                model_name,
                token=args.token,
                local_dir=local_dir,
                ignore_patterns=["*.pth", ".huggingface/*", "consolidated*"],
                tqdm_class=tqdm
            )

