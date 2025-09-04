# -*- coding: utf-8 -*-
"""
Experiment runner module for the ModelsTester project.

This module orchestrates the selection of datasets, targets, features and models,
and executes a training run. It uses the registry to discover available
functions, storage to persist computed features/targets, and history to
avoid duplicate runs. The aim is to provide a simple entry point for
sampling and evaluating combinations of components.
"""
from __future__ import annotations

import random
import uuid
from typing import List, Dict, Any, Optional
import os

import pandas as pd

from ..core import history  # run history management
from ..core.dedupe import make_run_key  # for computing unique run keys
from ..core.affinity import score_pair  # for feature-target correlation (optional)

from .. import storage  # data persistence
from ..registry import registry  # component registry


def select_components(random_state: Optional[random.Random] = None) -> Dict[str, Any]:
    """Randomly selects a target, a list of features and a model from the registry.

    You can override this function to implement smarter selection strategies
    (e.g. weighted by previous performance).

    Args:
        random_state: optional random.Random instance for reproducibility.

    Returns:
        Dict with keys: 'target', 'features', 'model', 'eval_cfg'
    """
    rng = random_state or random
    target_id = rng.choice(list(registry.targets.keys()))
    # choose between 1 and 3 features at random
    num_feats = rng.randint(1, min(3, len(registry.features)))
    feature_ids = rng.sample(list(registry.features.keys()), k=num_feats)
    model_id = rng.choice(list(registry.models.keys()))
    eval_cfg_id = rng.choice(list(registry.eval_cfgs.keys())) if registry.eval_cfgs else 'default'
    return {
        "target": target_id,
        "features": feature_ids,
        "model": model_id,
        "eval_cfg": eval_cfg_id,
    }


def run_experiment(
    df_name: str,
    split_id: str,
    seed: int = 0,
    db_path: str = "runs.db",
    random_state: Optional[random.Random] = None,
) -> str:
    """Runs a single experiment on the specified dataframe and split.

    This function performs the following steps:
      1. Select a target, features and model.
      2. Compute a run key and check if it already exists.
      3. Load or compute the target/feature series and persist them.
      4. Fit the model and compute metrics (placeholder).
      5. Record the run in the history database.

    Args:
        df_name: logical name of the dataset.
        split_id: identifier for the train/test split.
        seed: random seed to include in the run key.
        db_path: path to the SQLite history database.
        random_state: optional random.Random instance.

    Returns:
        The run_id of the executed experiment, or None if skipped.
    """
    rng = random_state or random

    # connect to history DB
    con = history.connect(db_path)
    history.init_db(con)

    # select components
    selection = select_components(rng)
    target_id = selection["target"]
    feature_ids = selection["features"]
    model_id = selection["model"]
    eval_cfg_id = selection["eval_cfg"]

    # Default leakage guard: drop features that look like predictions of the same target
    # Policy: only block if feature id starts with 'pred_<target>'
    if os.environ.get("DISABLE_TGT_FEAT_LEAK", "").lower() not in ("1", "true", "yes"):
        _leak_prefix = f"pred_{target_id}"
        feature_ids = [fid for fid in feature_ids if not str(fid).startswith(_leak_prefix)]
    # compute run key after guarding the feature list
    run_key = make_run_key(df_name, split_id, target_id, feature_ids, model_id, eval_cfg_id, seed)

    # check for duplicate
    if history.exists(con, run_key):
        try:
            con.close()
        except Exception:
            pass
        return None  # skip duplicated run

    # create run entry
    run_id = history.create_run(con, run_key, selection)

    try:
        # load dataframe
        df = storage.load_dataframe(df_name)

        # compute or load target
        if storage.target_exists(df_name, target_id):
            y = storage.load_target(df_name, target_id)
        else:
            y = registry.targets[target_id](df)
            storage.save_target(df_name, target_id, y)

        # compute or load features with skip-on-error
        features = {}
        failed_features = []
        for fid in feature_ids:
            try:
                if storage.feature_exists(df_name, fid):
                    x = storage.load_feature(df_name, fid)
                else:
                    x = registry.features[fid](df)
                    storage.save_feature(df_name, fid, x)
                features[fid] = x
            except Exception as fe:
                failed_features.append({"id": fid, "error": str(fe)})

        # optional: compute affinity scores for each feature with target (on training split)
        affinity_scores = {}
        for fid, x in features.items():
            try:
                affinity_scores[fid] = score_pair(y, x)
            except Exception:
                affinity_scores[fid] = {}

        # Fit the model (placeholder). Actual implementation depends on your models.
        model_fn = registry.models[model_id]
        # If no features succeeded, fail early but record failures
        if not features:
            raise RuntimeError("No features computed successfully")
        model, metrics = model_fn(y, features, df, selection)

        # finish run with success
        history.finish_run(con, run_id, "SUCCESS", metrics=metrics, artifacts={"affinity": affinity_scores, "failed_features": failed_features})
    except Exception as e:
        # record failure
        # attach failed feature info if available
        try:
            artifacts = {"failed_features": failed_features}
        except Exception:
            artifacts = None
        history.finish_run(con, run_id, "FAILED", metrics=None, artifacts=artifacts, err=str(e))
        raise
    finally:
        try:
            con.close()
        except Exception:
            pass

    return run_id


def run_experiment_with_selection(
    df_name: str,
    split_id: str,
    selection: Dict[str, Any],
    seed: int = 0,
    db_path: str = "runs.db",
    random_state: Optional[random.Random] = None,
) -> Optional[str]:
    """Run an experiment with an explicit component selection.

    Uses the same flow as run_experiment, but without random selection. Returns run_id or None if duplicate.
    """
    rng = random_state or random

    con = history.connect(db_path)
    history.init_db(con)

    target_id = selection["target"]
    feature_ids = selection["features"]
    model_id = selection["model"]
    eval_cfg_id = selection.get("eval_cfg", "default")

    # Default leakage guard: drop features that look like predictions of the same target
    # Policy: only block if feature id starts with 'pred_<target>'
    if os.environ.get("DISABLE_TGT_FEAT_LEAK", "").lower() not in ("1", "true", "yes"):
        _leak_prefix = f"pred_{target_id}"
        feature_ids = [fid for fid in feature_ids if not str(fid).startswith(_leak_prefix)]
    # compute run key after guarding the feature list
    run_key = make_run_key(df_name, split_id, target_id, feature_ids, model_id, eval_cfg_id, seed)
    if history.exists(con, run_key):
        try:
            con.close()
        except Exception:
            pass
        return None
    run_id = history.create_run(con, run_key, selection)

    try:
        df = storage.load_dataframe(df_name)

        if storage.target_exists(df_name, target_id):
            y = storage.load_target(df_name, target_id)
        else:
            y = registry.targets[target_id](df)
            storage.save_target(df_name, target_id, y)

        features = {}
        failed_features = []
        for fid in feature_ids:
            try:
                if storage.feature_exists(df_name, fid):
                    x = storage.load_feature(df_name, fid)
                else:
                    x = registry.features[fid](df)
                    storage.save_feature(df_name, fid, x)
                features[fid] = x
            except Exception as fe:
                failed_features.append({"id": fid, "error": str(fe)})

        affinity_scores = {}
        for fid, x in features.items():
            try:
                affinity_scores[fid] = score_pair(y, x)
            except Exception:
                affinity_scores[fid] = {}

        model_fn = registry.models[model_id]
        if not features:
            raise RuntimeError("No features computed successfully")
        model, metrics = model_fn(y, features, df, selection)

        history.finish_run(con, run_id, "SUCCESS", metrics=metrics, artifacts={"affinity": affinity_scores, "failed_features": failed_features})
    except Exception as e:
        try:
            artifacts = {"failed_features": failed_features}
        except Exception:
            artifacts = None
        history.finish_run(con, run_id, "FAILED", metrics=None, artifacts=artifacts, err=str(e))
        raise
    finally:
        try:
            con.close()
        except Exception:
            pass

    return run_id
