# =============================================================
# engine/evaluator.py
# Evaluator for the COD Hybrid model.
#
# Phase 1: Basic placeholder — runs inference and reports
#          average BCE loss on the test set so you can confirm
#          the pipeline works end-to-end.
#
# Phase 2: Full implementation with standard COD metrics:
#   - MAE   (Mean Absolute Error)
#   - Sm    (S-measure)   — structural similarity
#   - Em    (E-measure)   — enhanced alignment measure
#   - Fm    (F-measure)   — weighted F-beta score
#
# These four metrics are the standard COD benchmark numbers
# reported in ZoomNeXt Table 1 and all published COD papers.
# =============================================================

import torch
import configs.cod_hybrid as cfg
from losses import BCELoss


class Evaluator:
    """
    Runs inference on a test DataLoader and computes metrics.

    Phase 1:
        Reports average BCE loss on the test set.
        Confirms the full pipeline (data → model → loss) works.

    Phase 2:
        Computes MAE, S-measure, E-measure, F-measure per dataset.

    Args:
        model  : CODHybrid model instance
        device : torch.device

    Usage:
        evaluator = Evaluator(model, device)
        evaluator.evaluate(test_loader, dataset_name='CAMO')
    """

    def __init__(self, model, device):
        self.model    = model
        self.device   = device
        self.bce_loss = BCELoss()

    @torch.no_grad()
    def evaluate(self, loader, dataset_name: str = '') -> dict:
        """
        Run evaluation on a single test DataLoader.

        Phase 1: computes average BCE loss only.

        Args:
            loader       : DataLoader from build_eval_loader()
            dataset_name : name of dataset being evaluated (for logging)

        Returns:
            metrics : dict with evaluation results
                Phase 1: {'bce': float}
                Phase 2: {'bce': float, 'mae': float,
                           'Sm': float, 'Em': float, 'Fm': float}
        """
        self.model.eval()

        total_bce = 0.0
        n_batches = 0

        for batch in loader:
            image = batch['image'].to(self.device)  # [1, 3, H, W]
            mask  = batch['mask'].to(self.device)   # [1, 1, H, W]

            # ── Forward pass ─────────────────────────────────
            outputs = self.model(image)
            pred    = outputs['pred']               # [1, 1, H, W]

            # ── Phase 1: BCE loss only ───────────────────────
            loss_bce   = self.bce_loss(pred, mask)
            total_bce += loss_bce.item()
            n_batches += 1

            # ── Phase 2: metric computation goes here ─────────
            # Steps to implement in Phase 2:
            #
            # 1. Convert pred to numpy, threshold at 0.5 → binary mask
            # 2. MAE:
            #       mae = |pred - mask|.mean()
            # 3. S-measure (Sm):
            #       Combines object-aware (So) and region-aware (Sr)
            #       Sm = α * So + (1-α) * Sr  where α=0.5
            # 4. E-measure (Em):
            #       Enhanced alignment between pred and GT
            #       Em = (2 * μ_fg * μ_gt) / (μ_fg^2 + μ_gt^2 + eps)
            # 5. F-measure (Fm):
            #       Fm = (1+β²) * precision * recall / (β²*precision + recall)
            #       where β²=0.3 (standard COD setting)

        avg_bce = total_bce / max(n_batches, 1)

        metrics = {'bce': avg_bce}

        # ── Log results ──────────────────────────────────────
        print(f'\n[evaluator] {dataset_name} results:')
        print(f'  BCE loss : {avg_bce:.4f}')
        print(f'  (Full metrics — MAE, Sm, Em, Fm — active in Phase 2)\n')

        return metrics

    @torch.no_grad()
    def evaluate_all(self, loaders: dict) -> dict:
        """
        Evaluate on all datasets in the loaders dict.

        Args:
            loaders : dict from build_all_eval_loaders()
                      {'CAMO': loader, 'COD10K': loader}

        Returns:
            all_metrics : dict {'CAMO': {...}, 'COD10K': {...}}
        """
        all_metrics = {}
        for name, loader in loaders.items():
            all_metrics[name] = self.evaluate(loader, dataset_name=name)
        return all_metrics
