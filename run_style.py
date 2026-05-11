import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Neural style transfer with VGG19.")
    parser.add_argument("--content", default="data/content", help="content image file or directory")
    parser.add_argument("--style", default="data/style", help="style image file or directory")
    parser.add_argument("--out", default="outputs/result.png", help="output image path")
    parser.add_argument("--image-size", type=int, default=384, help="square resize size")
    parser.add_argument("--steps", type=int, default=300, help="optimization steps")
    parser.add_argument("--lr", type=float, default=0.03, help="Adam learning rate")
    parser.add_argument("--content-weight", type=float, default=1.0)
    parser.add_argument("--style-weight", type=float, default=500000.0)
    parser.add_argument("--content-layer", default="relu4_2")
    parser.add_argument(
        "--style-layers",
        nargs="+",
        default=["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"],
    )
    parser.add_argument("--init", choices=["content", "noise"], default="content")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--vgg-weights", default=None, help="optional local VGG19 .pth file")
    parser.add_argument("--no-pretrained", action="store_true", help="debug only; output quality will be poor")
    parser.add_argument("--log-interval", type=int, default=25)
    return parser.parse_args()


def main():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is not installed. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    from libs.dataset import resolve_single_image
    from libs.image_utils import choose_device, load_image, save_image
    from libs.style_loss import StyleTransferObjective
    from libs.vgg_features import build_vgg19

    args = parse_args()
    device = choose_device(args.device)

    content_path = resolve_single_image(args.content)
    style_path = resolve_single_image(args.style)
    output_path = Path(args.out)

    print(f"Device: {device}")
    print(f"Content: {content_path}")
    print(f"Style: {style_path}")

    content_image = load_image(content_path, device=device, image_size=args.image_size)
    style_image = load_image(style_path, device=device, image_size=args.image_size)

    output_layers = sorted(set([args.content_layer] + args.style_layers))
    extractor = build_vgg19(
        output_layers=output_layers,
        device=device,
        weights_path=args.vgg_weights,
        pretrained=not args.no_pretrained,
    )

    objective = StyleTransferObjective(
        extractor=extractor,
        content_image=content_image,
        style_image=style_image,
        content_layer=args.content_layer,
        style_layers=args.style_layers,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
    )

    if args.init == "content":
        generated = content_image.clone()
    else:
        generated = torch.rand_like(content_image)
    generated.requires_grad_(True)

    optimizer = torch.optim.Adam([generated], lr=args.lr)

    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        total_loss, content_loss, style_loss = objective(generated)
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            generated.clamp_(0.0, 1.0)

        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            print(
                f"step {step:04d}/{args.steps} "
                f"total={total_loss.item():.4f} "
                f"content={content_loss.item():.4f} "
                f"style={style_loss.item():.4f}"
            )

    save_image(generated, output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

