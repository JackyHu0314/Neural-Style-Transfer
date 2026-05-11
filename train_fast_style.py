import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train a feed-forward style-transfer model.")
    parser.add_argument("--content-dir", default="data/train/content/val2017")
    parser.add_argument("--style", default="data/style/starry_night.jpg")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--sample-dir", default="outputs/training_samples")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--content-weight", type=float, default=1e5)
    parser.add_argument("--style-weight", type=float, default=1e10)
    parser.add_argument("--tv-weight", type=float, default=1e-6)
    parser.add_argument("--content-layer", default="relu3_3")
    parser.add_argument(
        "--style-layers",
        nargs="+",
        default=["relu1_1", "relu2_1", "relu3_1", "relu4_1"],
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--vgg-weights", default=None)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--sample-interval", type=int, default=200)
    parser.add_argument("--limit-images", type=int, default=None, help="debug subset size")
    return parser.parse_args()


def main():
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset

    from libs.image_utils import choose_device, load_image, normalize_batch, save_image
    from libs.style_loss import gram_matrix
    from libs.train_dataset import ContentImageDataset
    from libs.transform_net import TransformerNet, total_variation_loss
    from libs.vgg_features import build_vgg19

    args = parse_args()
    device = choose_device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)
    sample_dir = Path(args.sample_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    dataset = ContentImageDataset(args.content_dir, image_size=args.image_size)
    if args.limit_images is not None:
        dataset = Subset(dataset, range(min(args.limit_images, len(dataset))))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    output_layers = sorted(set([args.content_layer] + args.style_layers))
    extractor = build_vgg19(output_layers, device=device, weights_path=args.vgg_weights)
    transformer = TransformerNet().to(device)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=args.lr)

    style_image = load_image(args.style, device=device, image_size=args.image_size)
    with torch.no_grad():
        style_features = extractor(normalize_batch(style_image))
        style_targets = {
            layer: gram_matrix(style_features[layer]).detach()
            for layer in args.style_layers
        }

    print(f"Device: {device}")
    print(f"Content images: {len(dataset)}")
    print(f"Style image: {args.style}")
    print(f"Checkpoint dir: {checkpoint_dir}")

    global_step = 0
    transformer.train()
    for epoch in range(1, args.epochs + 1):
        for batch in loader:
            global_step += 1
            content_images = batch.to(device, non_blocking=True)

            generated = transformer(content_images)

            generated_features = extractor(normalize_batch(generated))
            with torch.no_grad():
                content_features = extractor(normalize_batch(content_images))

            content_loss = F.mse_loss(
                generated_features[args.content_layer],
                content_features[args.content_layer],
            )

            style_loss = generated.new_tensor(0.0)
            for layer in args.style_layers:
                generated_gram = gram_matrix(generated_features[layer])
                target_gram = style_targets[layer].expand_as(generated_gram)
                style_loss = style_loss + F.mse_loss(generated_gram, target_gram)
            style_loss = style_loss / len(args.style_layers)

            tv_loss = total_variation_loss(generated)
            total_loss = (
                args.content_weight * content_loss
                + args.style_weight * style_loss
                + args.tv_weight * tv_loss
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if global_step == 1 or global_step % args.log_interval == 0:
                print(
                    f"epoch={epoch} step={global_step} "
                    f"total={total_loss.item():.2f} "
                    f"content={content_loss.item():.6f} "
                    f"style={style_loss.item():.6f} "
                    f"tv={tv_loss.item():.6f}"
                )

            if global_step == 1 or global_step % args.sample_interval == 0:
                save_image(generated[0:1], sample_dir / f"step_{global_step:06d}.png")

        checkpoint_path = checkpoint_dir / f"fast_style_epoch_{epoch}.pth"
        torch.save(
            {
                "model_state_dict": transformer.state_dict(),
                "style_image": args.style,
                "image_size": args.image_size,
                "content_layer": args.content_layer,
                "style_layers": args.style_layers,
                "content_weight": args.content_weight,
                "style_weight": args.style_weight,
                "tv_weight": args.tv_weight,
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()

