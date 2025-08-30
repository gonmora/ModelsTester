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
    df_version: str,
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
        df_version: version identifier for the dataset.
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

    # compute run key
    run_key = make_run_key(df_name, df_version, split_id, target_id, feature_ids, model_id, eval_cfg_id, seed)

    # check for duplicate
    if history.exists(con, run_key):
        return None  # skip duplicated run

    # create run entry
    run_id = history.create_run(con, run_key, selection)

    try:
        # load dataframe
        df = storage.load_dataframe(df_name, df_version)

        # compute or load target
        if storage.target_exists(df_name, df_version, target_id):
            y = storage.load_target(df_name, df_version, target_id)
        else:
            y = registry.targets[target_id](df)
            storage.save_target(y, df_name, df_version, target_id)

        # compute or load features
        features = {}
        for fid in feature_ids:
            if storage.feature_exists(df_name, df_version, fid):
                x = storage.load_feature(df_name, df_version, fid)
            else:
                x = registry.features[fid](df)
                storage.save_feature(x, df_name, df_version, fid)
            features[fid] = x

        # optional: compute affinity scores for each feature with target (on training split)
        affinity_scores = {}
        for fid, x in features.items():
            try:
                affinity_scores[fid] = score_pair(y, x)
            except Exception:
                affinity_scores[fid] = {}

        # Fit the model (placeholder). Actual implementation depends on your models.
        model_fn = registry.models[model_id]
        model, metrics = model_fn(y, features, df, selection)

        # finish run with success
        history.finish_run(con, run_id, "SUCCESS", metrics=metrics, artifacts={"affinity": affinity_scores})
    except Exception as e:
        # record failure
        history.finish_run(con, run_id, "FAILED", metrics=None, artifacts=None, err=str(e))
        raise

    return run_id
