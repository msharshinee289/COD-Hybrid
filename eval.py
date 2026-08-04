# =============================================================
# eval.py
# Entry point for evaluating the COD Hybrid model.
#
# Usage:
#   # Evaluate on all datasets (CAMO + COD10K)
#   python eval.py --checkpoint checkpoints/ckpt_epoch_100.pth
#
#   # Evaluate on a single dataset
#   python eval.py --checkpoint checkpoints/ckpt_epoch_100.pth
#                  --dataset CAMO
#
#   # Evaluate on CPU
#   python eval.py --checkpoint checkpoints/ckpt_epoch_100.pth
#                  --device cpu
# =============================================================

import argparse
import torch

import configs.cod_hybrid as cfg
from utils.misc        import get_device
from models            import build_model
from data.build        import build_eval_loader, build_all_eval_loaders
from engine.evaluator  import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate COD Hybrid model'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.pth file)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Dataset to evaluate on: CAMO or COD10K. '
             'If not specified, evaluates on all datasets in config.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=cfg.DEVICE,
        help='Device: "cuda" or "cpu"'
    )
    return parser.parse_args()


def main():
    args   = parse_args()
    device = get_device(args.device)

    # ── Model ────────────────────────────────────────────────
    print('\n[eval] Building model...')
    model = build_model(pretrained=False)
    # pretrained=False — we load our own trained weights below

    # ── Load checkpoint ──────────────────────────────────────
    print(f'[eval] Loading checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    print(f'[eval] Loaded weights from epoch {ckpt.get("epoch", "?")}')

    model = model.to(device)

    # ── Evaluator ────────────────────────────────────────────
    evaluator = Evaluator(model=model, device=device)

    # ── Run evaluation ───────────────────────────────────────
    if args.dataset is not None:
        # Single dataset
        loader  = build_eval_loader(args.dataset)
        metrics = evaluator.evaluate(loader, dataset_name=args.dataset)
    else:
        # All datasets
        loaders     = build_all_eval_loaders()
        all_metrics = evaluator.evaluate_all(loaders)

        # ── Summary table ────────────────────────────────────
        print('\n[eval] ── Summary ──────────────────────────')
        for dataset, metrics in all_metrics.items():
            bce = metrics.get('bce', float('nan'))
            print(f'  {dataset:<10}  BCE: {bce:.4f}')
            # Phase 2: also print MAE, Sm, Em, Fm here
        print('[eval] ────────────────────────────────────\n')


if __name__ == '__main__':
    main()
