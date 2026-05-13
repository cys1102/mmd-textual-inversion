import torch
import os
import glob
import argparse
from itertools import product
from tqdm import trange

# DEFAULT_EMBED_PATH = "{dataset}-tokens-10way-per-image/{dataset}-{seed}-{examples_per_class}.pt"
DEFAULT_EMBED_PATH = "{dataset}-tokens-mmd{dist_match}/{dataset}-{seed}-{examples_per_class}.pt"
DISC_EMBED_PATH = "{dataset}-tokens-tuned-mmd{dist_match}/{dataset}-{seed}-{examples_per_class}.pt"


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Merge token files")

    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--num-way", type=int, default=3)
    parser.add_argument("--examples-per-class", nargs='+', type=int, default=[1, 2, 4, 8, 16])
    
    parser.add_argument("--embed-path", type=str, default=DEFAULT_EMBED_PATH)
    parser.add_argument("--input-path", type=str, default="./fine-tuned-mmd{dist_match}")
    parser.add_argument("--disc", action="store_true", default=False)
    parser.add_argument("--weight", type=float, default=-1.0)
    parser.add_argument("--dataset", type=str, default="pets", 
                        choices=["aircraft", "dtd", "coco", "pascal", "pets", "cub", "flowers"])
    parser.add_argument("--dist_match", type=float, default=0.005)
    

    args = parser.parse_args()
    if args.disc:
        args.embed_path = DISC_EMBED_PATH

    args.input_path = args.input_path.format(dist_match=args.dist_match)
    print(f"Input path: {args.input_path}")
    for seed, examples_per_class in product(
            range(args.num_trials), args.examples_per_class):
        # seed += 5
        # path = os.path.join(args.input_path, (
        #     f"{args.dataset}-{seed}-{examples_per_class}/*/*/learned_embeds.bin"))
        path = os.path.join(args.input_path, (
            f"{args.dataset}-{seed}-{examples_per_class}/*/learned_embeds.bin"))
        print(f"Input path is: {path}")

        merged_dict = dict()
        for file in glob.glob(path):
            print(file)
            merged_dict.update(torch.load(file))
        if not merged_dict:
            print(f"Skipping {path}")
            continue
        # target_path = args.embed_path.format(
        #     dataset=args.dataset, seed=seed, 
        #     examples_per_class=examples_per_class) if not args.disc else args.embed_path.format(dataset=args.dataset, weight=args.weight, seed=seed, examples_per_class=examples_per_class)
        target_path = args.embed_path.format(
            dataset=args.dataset, seed=seed, dist_match=args.dist_match,
            examples_per_class=examples_per_class)
        print(f"Target path is: {target_path}")
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        torch.save(merged_dict, target_path)