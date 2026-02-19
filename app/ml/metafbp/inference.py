"""MetaFBP production inference wrapper for Harmonia V3.

Loads trained checkpoints (Stage 1 extractor + Stage 2 meta-generator) and
provides methods for per-user adaptation and scoring of target images.

Usage:
    from app.ml.metafbp.inference import MetaFBPInference

    engine = MetaFBPInference(
        extractor_path="models/universal_extractor.pth",
        generator_path="models/meta_generator.pth",
    )

    # Adapt to a user's preferences
    adapted = engine.adapt_user_predictor(
        support_images=["img1.jpg", "img2.jpg", ...],
        support_ratings=[5, 2, 4, 1, 3, ...],
    )

    # Score a target image
    score = engine.score_target(adapted, "target.jpg")
    # score is in 0-100 range
"""

from __future__ import annotations

import io
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import torch
import torch.nn.functional as F

from app.ml.metafbp.learner import Learner
from app.ml.metafbp.meta import Meta
from app.ml.metafbp.preprocessing import (
    inference_transform,
    preprocess_batch,
    preprocess_image,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MODEL_SCORE_MIN = 1.0
_MODEL_SCORE_MAX = 5.0
_OUTPUT_SCORE_MIN = 0.0
_OUTPUT_SCORE_MAX = 100.0


class MetaFBPInference:
    """Production inference wrapper for the MetaFBP personalized beauty model.

    Loads:
        * Stage 1 checkpoint -- frozen ResNet-18 feature extractor (``Learner``)
        * Stage 2 checkpoint -- meta-generator base weights (``Meta``)

    Exposes two public methods:
        * ``adapt_user_predictor`` -- inner-loop adaptation on a support set
        * ``score_target``         -- score a single target image with adapted weights
    """

    def __init__(
        self,
        extractor_path: str = "models/universal_extractor.pth",
        generator_path: str = "models/meta_generator.pth",
        feature_dim: int = 512,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
        adaptation_lambda: float = 0.01,
        device: str | None = None,
    ) -> None:
        """Initialise MetaFBP inference engine.

        Args:
            extractor_path: Path to Stage 1 checkpoint (frozen ResNet-18).
            generator_path: Path to Stage 2 checkpoint (Meta generator + predictor).
            feature_dim: Dimension of the feature embeddings (default 512).
            inner_lr: Inner-loop learning rate alpha (default 0.01).
            inner_steps: Number of inner-loop SGD steps k (default 5).
            adaptation_lambda: Adaptation strength lambda (default 0.01).
            device: Force a device ("cpu" / "cuda"). If ``None``, auto-detect.
        """
        # ----- Device ------------------------------------------------------ #
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        logger.info(
            "metafbp.init",
            device=str(self.device),
            feature_dim=feature_dim,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
            adaptation_lambda=adaptation_lambda,
        )

        self.feature_dim = feature_dim
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.adaptation_lambda = adaptation_lambda

        # ----- Stage 1: Feature Extractor (Learner) ------------------------ #
        self.extractor = Learner()
        self._load_extractor(extractor_path)
        self.extractor.to(self.device)
        self.extractor.eval()
        # Freeze extractor -- no gradient computation needed
        for param in self.extractor.vars:
            param.requires_grad_(False)
        for param in self.extractor.vars_bn:
            param.requires_grad_(False)

        # ----- Stage 2: Meta-learner (Generator + Predictor) --------------- #
        self.meta = Meta(feature_dim=feature_dim)
        self._load_generator(generator_path)
        self.meta.to(self.device)
        self.meta.eval()

        logger.info("metafbp.ready")

    # ------------------------------------------------------------------ #
    # Checkpoint loading
    # ------------------------------------------------------------------ #
    def _load_extractor(self, path: str) -> None:
        """Load Stage 1 feature-extractor checkpoint.

        Handles both full model state dicts and ``Learner``-specific
        ``vars.*`` / ``vars_bn.*`` key patterns.
        """
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            logger.warning(
                "metafbp.extractor_checkpoint_missing",
                path=str(ckpt_path),
                msg="Using randomly initialised extractor -- predictions will be meaningless.",
            )
            return

        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            # Handle wrapped checkpoints (e.g. {"model_state_dict": {...}})
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            self.extractor.load_state_dict(state)
            logger.info("metafbp.extractor_loaded", path=str(ckpt_path))
        except Exception:
            logger.exception("metafbp.extractor_load_failed", path=str(ckpt_path))
            raise

    def _load_generator(self, path: str) -> None:
        """Load Stage 2 meta-generator checkpoint.

        Handles both full model state dicts and ``Meta``-specific key patterns.
        """
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            logger.warning(
                "metafbp.generator_checkpoint_missing",
                path=str(ckpt_path),
                msg="Using randomly initialised generator -- predictions will be meaningless.",
            )
            return

        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            self.meta.load_state_dict(state)
            logger.info("metafbp.generator_loaded", path=str(ckpt_path))
        except Exception:
            logger.exception("metafbp.generator_load_failed", path=str(ckpt_path))
            raise

    # ------------------------------------------------------------------ #
    # Feature extraction
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _extract_features(self, image_path: str) -> torch.Tensor:
        """Extract a 512-dim feature vector from a single image.

        Returns:
            Tensor of shape ``(1, 512)`` on ``self.device``.
        """
        tensor = preprocess_image(image_path).to(self.device)
        features = self.extractor(tensor, bn_training=False)
        return features

    @torch.no_grad()
    def _extract_features_batch(self, image_paths: list[str]) -> torch.Tensor:
        """Extract features for a batch of images.

        Returns:
            Tensor of shape ``(N, 512)`` on ``self.device``.
        """
        batch = preprocess_batch(image_paths).to(self.device)
        features = self.extractor(batch, bn_training=False)
        return features

    # ------------------------------------------------------------------ #
    # Cyclical resampling for imbalanced support sets
    # ------------------------------------------------------------------ #
    @staticmethod
    def _cyclical_resample(
        ratings: list[int],
        min_per_class: int = 2,
    ) -> tuple[list[int], list[int]]:
        """Resample indices so every observed rating class has at least
        ``min_per_class`` examples via cyclical duplication.

        This prevents inner-loop adaptation from being dominated by a single
        rating bucket (e.g. all 5s with a single 1).

        Args:
            ratings: Original list of integer ratings (1-5).
            min_per_class: Minimum samples required per observed class.

        Returns:
            Tuple of ``(resampled_indices, resampled_ratings)`` where indices
            refer back to the original list positions.
        """
        class_indices: dict[int, list[int]] = {}
        for idx, r in enumerate(ratings):
            class_indices.setdefault(r, []).append(idx)

        resampled_indices: list[int] = []
        for rating_class, indices in class_indices.items():
            if len(indices) >= min_per_class:
                resampled_indices.extend(indices)
            else:
                # Cyclically duplicate until we reach min_per_class
                extended = indices.copy()
                cycle_idx = 0
                while len(extended) < min_per_class:
                    extended.append(indices[cycle_idx % len(indices)])
                    cycle_idx += 1
                resampled_indices.extend(extended)

        resampled_ratings = [ratings[i] for i in resampled_indices]
        return resampled_indices, resampled_ratings

    # ------------------------------------------------------------------ #
    # User adaptation (public API)
    # ------------------------------------------------------------------ #
    def adapt_user_predictor(
        self,
        support_images: list[str],
        support_ratings: list[int],
    ) -> dict[str, Any]:
        """Run k-step inner-loop SGD adaptation on a user's support set.

        This produces a personalised set of predictor weights that encode the
        user's individual aesthetic preferences.  The returned state dict can
        be serialised (e.g. to Redis or Postgres JSONB) and re-loaded later
        for scoring without re-running adaptation.

        Args:
            support_images: List of file paths to the user's rated face images.
            support_ratings: Corresponding integer ratings (1-5) for each image.

        Returns:
            A dict containing the adapted predictor state:
                - ``adapted_weight``: Tensor of shape ``(1, 512)``
                - ``adapted_bias``: Tensor of shape ``(1,)``
                - ``dynamic_weight``: Tensor of shape ``(1, 512)``
                - ``num_support``: Number of support samples (after resampling)
                - ``rating_distribution``: Counter of ratings used

        Raises:
            ValueError: If support_images and support_ratings have different lengths
                or if either is empty.
        """
        if len(support_images) != len(support_ratings):
            raise ValueError(
                f"support_images ({len(support_images)}) and support_ratings "
                f"({len(support_ratings)}) must have the same length."
            )
        if not support_images:
            raise ValueError("At least one support image is required.")

        logger.info(
            "metafbp.adapt_start",
            num_images=len(support_images),
            rating_dist=dict(Counter(support_ratings)),
        )

        # Cyclical resampling for class balance
        resampled_indices, resampled_ratings = self._cyclical_resample(
            support_ratings, min_per_class=2
        )

        # Re-order images to match resampled indices (may have duplicates)
        resampled_images = [support_images[i] for i in resampled_indices]

        # Extract features for the (resampled) support set
        support_features = self._extract_features_batch(resampled_images)
        support_labels = torch.tensor(
            resampled_ratings, dtype=torch.float32, device=self.device
        )

        # ----- Inner-loop SGD adaptation (k steps) ---------------------- #
        weight = self.meta.predictor_weight.clone().detach().requires_grad_(True)
        bias = self.meta.predictor_bias.clone().detach().requires_grad_(True)

        for step in range(self.inner_steps):
            pred = F.linear(support_features, weight, bias)
            loss = F.mse_loss(pred.squeeze(), support_labels)

            grad_w, grad_b = torch.autograd.grad(loss, [weight, bias])
            # Detach and re-enable grad for next step
            weight = (weight - self.inner_lr * grad_w).detach().requires_grad_(True)
            bias = (bias - self.inner_lr * grad_b).detach().requires_grad_(True)

            logger.debug(
                "metafbp.adapt_step",
                step=step + 1,
                loss=loss.item(),
            )

        # ----- Generate dynamic weight via Parameter Generator ----------- #
        with torch.no_grad():
            dynamic_weight = self.meta.generate_dynamic_weights(support_features)

        adapted_state = {
            "adapted_weight": weight.detach().cpu(),
            "adapted_bias": bias.detach().cpu(),
            "dynamic_weight": dynamic_weight.detach().cpu(),
            "num_support": len(resampled_images),
            "rating_distribution": dict(Counter(resampled_ratings)),
        }

        logger.info(
            "metafbp.adapt_complete",
            num_support=adapted_state["num_support"],
            rating_dist=adapted_state["rating_distribution"],
        )

        return adapted_state

    # ------------------------------------------------------------------ #
    # Target scoring (public API)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def score_target(
        self,
        adapted_state: dict[str, Any],
        target_image_path: str,
    ) -> float:
        """Score a single target image using adapted predictor weights.

        Pipeline:
            1. Extract 512-dim features from the target image via the frozen
               ResNet-18 extractor.
            2. Combine adapted weights with the dynamic Parameter Generator
               output using adaptation strength lambda.
            3. Compute the raw predicted score (model range ~1-5).
            4. Normalise to the 0-100 Harmonia score range.

        Args:
            adapted_state: Dict returned by ``adapt_user_predictor``.
            target_image_path: File path to the target face image.

        Returns:
            A float in [0, 100] representing the personalised attractiveness
            score.

        Raises:
            KeyError: If adapted_state is missing required keys.
            FileNotFoundError: If the target image does not exist.
        """
        # Validate adapted state
        required_keys = {"adapted_weight", "adapted_bias", "dynamic_weight"}
        missing = required_keys - set(adapted_state.keys())
        if missing:
            raise KeyError(
                f"adapted_state is missing required keys: {missing}"
            )

        target_path = Path(target_image_path)
        if not target_path.exists():
            raise FileNotFoundError(
                f"Target image not found: {target_image_path}"
            )

        # Move adapted tensors to device
        adapted_weight = adapted_state["adapted_weight"].to(self.device)
        adapted_bias = adapted_state["adapted_bias"].to(self.device)
        dynamic_weight = adapted_state["dynamic_weight"].to(self.device)

        # 1. Extract target features
        target_features = self._extract_features(target_image_path)

        # 2. Apply adaptation strength: theta_f_dynamic = theta'_f + lambda * G(x)
        final_weight = adapted_weight + self.adaptation_lambda * dynamic_weight

        # 3. Compute raw prediction
        raw_prediction = F.linear(target_features, final_weight, adapted_bias)
        raw_score = raw_prediction.squeeze().item()

        # 4. Clamp to model range [1, 5] and normalise to [0, 100]
        clamped = max(_MODEL_SCORE_MIN, min(_MODEL_SCORE_MAX, raw_score))
        normalised = (clamped - _MODEL_SCORE_MIN) * (
            _OUTPUT_SCORE_MAX / (_MODEL_SCORE_MAX - _MODEL_SCORE_MIN)
        )
        # Equivalent to: (clamped - 1) * 25

        logger.info(
            "metafbp.score",
            target=target_image_path,
            raw_score=round(raw_score, 4),
            clamped=round(clamped, 4),
            normalised=round(normalised, 2),
        )

        return round(normalised, 2)

    # ------------------------------------------------------------------ #
    # Batch scoring convenience
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def score_batch(
        self,
        adapted_state: dict[str, Any],
        target_image_paths: list[str],
    ) -> list[float]:
        """Score multiple target images with the same adapted weights.

        Args:
            adapted_state: Dict returned by ``adapt_user_predictor``.
            target_image_paths: List of file paths to target face images.

        Returns:
            List of floats in [0, 100], one per target image.
        """
        return [
            self.score_target(adapted_state, path) for path in target_image_paths
        ]
