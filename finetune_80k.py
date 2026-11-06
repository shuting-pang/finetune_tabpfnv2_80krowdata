"""Fine-tune TabPFN-2 across 80,000 Covertype training rows via resampling.

Requires the legacy fine-tuning API in tabpfn==2.1.3. A GPU is strongly
recommended. No private data or model weights are bundled with this repository.
"""

from __future__ import annotations

import argparse
import csv
import json
from functools import partial
from pathlib import Path

import numpy as np
from resampling import epoch_batches
from sklearn.datasets import fetch_covtype
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-rows", type=int, default=80_000)
    parser.add_argument("--test-rows", type=int, default=20_000)
    parser.add_argument("--context-rows", type=int, default=10_000)
    parser.add_argument("--query-fraction", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-home", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cover80k"))
    parser.add_argument("--model-path", type=Path, default=None,
                        help="Optional local TabPFN-2 classifier checkpoint")
    parser.add_argument("--no-download", action="store_true",
                        help="Require Covertype to be present in --data-home")
    args = parser.parse_args()
    if args.train_rows < 2 or args.test_rows < 1:
        parser.error("--train-rows must be >=2 and --test-rows must be >=1")
    if not 2 <= args.context_rows <= 10_000:
        parser.error("--context-rows must be between 2 and 10000 for TabPFN-2")
    if not 0 < args.query_fraction < 1:
        parser.error("--query-fraction must be in (0, 1)")
    if args.epochs < 1 or args.learning_rate <= 0:
        parser.error("--epochs and --learning-rate must be positive")
    return args


def load_cover_split(args: argparse.Namespace) -> tuple[np.ndarray, ...]:
    X_all, y_all = fetch_covtype(
        data_home=str(args.data_home) if args.data_home else None,
        download_if_missing=not args.no_download,
        return_X_y=True,
    )
    total = args.train_rows + args.test_rows
    if total > len(y_all):
        raise ValueError(f"Requested {total} rows; Covertype has {len(y_all)}")
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(len(y_all), size=total, replace=False)
    X, y = X_all[chosen], y_all[chosen]
    return train_test_split(
        X, y, train_size=args.train_rows, test_size=args.test_rows,
        stratify=y, random_state=args.seed,
    )


def main() -> None:
    args = parse_args()
    try:
        import tabpfn
        import torch
        from torch.optim import Adam
        from torch.utils.data import DataLoader
        from tabpfn import TabPFNClassifier
        from tabpfn.finetune_utils import clone_model_for_evaluation
        from tabpfn.model_loading import save_tabpfn_model
        from tabpfn.utils import meta_dataset_collator
    except (ImportError, OSError) as exc:
        raise SystemExit(
            "Install the pinned requirements in a working CUDA/PyTorch environment. "
            f"Import failed: {exc}"
        ) from exc

    if tabpfn.__version__ != "2.1.3":
        raise SystemExit(f"Expected tabpfn==2.1.3, found {tabpfn.__version__}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for this 80k fine-tuning example")

    X_train, X_test, y_train, y_test = load_cover_split(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["data_home"] = str(args.data_home) if args.data_home else None
    config["output_dir"] = str(args.output_dir)
    config["model_path"] = str(args.model_path) if args.model_path else None
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Training rows: {len(y_train)}; held-out test rows: {len(y_test)}")

    model_args = dict(
        device="cuda", n_estimators=2, random_state=args.seed,
        inference_precision=torch.float32, ignore_pretraining_limits=True,
        fit_mode="batched", differentiable_input=False,
    )
    if args.model_path:
        if not args.model_path.is_file():
            raise FileNotFoundError(args.model_path)
        model_args["model_path"] = str(args.model_path)
    classifier = TabPFNClassifier(**model_args)
    classifier._initialize_model_variables()
    optimizer = Adam(classifier.model_.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    eval_args = {**model_args, "fit_mode": "fit_preprocessors",
                 "inference_config": {"SUBSAMPLE_SAMPLES": args.context_rows}}
    eval_args.pop("model_path", None)  # clone supplies the trained model object

    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "roc_auc_ovr_weighted", "log_loss"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            # Resampling occurs here: every row appears exactly once per epoch,
            # while its 10k-row companions change with the epoch seed.
            indices = epoch_batches(len(y_train), args.context_rows, args.seed, epoch)
            X_groups = [X_train[idx] for idx in indices]
            y_groups = [y_train[idx] for idx in indices]
            split_fn = partial(
                train_test_split, test_size=args.query_fraction,
                random_state=args.seed + epoch, stratify=None,
            )
            datasets = classifier.get_preprocessed_datasets(
                X_groups, y_groups, split_fn, args.context_rows
            )
            loader = DataLoader(datasets, batch_size=1, collate_fn=meta_dataset_collator)
            classifier.model_.train()
            losses = []
            for X_context, X_query, y_context, y_query, cat_ix, confs in loader:
                context_labels = np.concatenate([v.cpu().numpy().ravel() for v in y_context])
                query_labels = y_query.cpu().numpy().ravel()
                if set(np.unique(context_labels)) != set(np.unique(query_labels)):
                    raise RuntimeError("A resampled batch lost a class in its context or query")
                optimizer.zero_grad(set_to_none=True)
                classifier.fit_from_preprocessed(X_context, y_context, cat_ix, confs)
                logits = classifier.forward(X_query, return_logits=True)
                loss = criterion(logits, y_query.to("cuda"))
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            eval_classifier = clone_model_for_evaluation(classifier, eval_args, TabPFNClassifier)
            eval_classifier.fit(X_train, y_train)
            probabilities = eval_classifier.predict_proba(X_test)
            row = dict(
                epoch=epoch, train_loss=float(np.mean(losses)),
                roc_auc_ovr_weighted=float(roc_auc_score(
                    y_test, probabilities, multi_class="ovr", average="weighted"
                )),
                log_loss=float(log_loss(y_test, probabilities)),
            )
            writer.writerow(row)
            fh.flush()
            print(row)

    # Save the fine-tuned foundation weights. In tabpfn 2.1.3,
    # save_fitted_tabpfn_model omits foundation weights, so it is unsuitable here.
    save_tabpfn_model(classifier, args.output_dir / "tabpfn_v2_80k_finetuned.ckpt")
    print(f"Saved model and metrics in {args.output_dir}")


if __name__ == "__main__":
    main()
