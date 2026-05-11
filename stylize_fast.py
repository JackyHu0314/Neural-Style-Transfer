import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Stylize an image with a trained model.")
    parser.add_argument("--input", default="data/content/city_street.jpg")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="outputs/fast_result.png")
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    import torch

    from libs.image_utils import choose_device, load_image, save_image
    from libs.transform_net import TransformerNet

    args = parse_args()
    device = choose_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = TransformerNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = load_image(args.input, device=device, max_size=args.max_size)
    with torch.no_grad():
        output = model(image)

    out_path = Path(args.out)
    save_image(output, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

