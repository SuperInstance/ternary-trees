# ternary-trees

Decision trees and random forests for ternary classification on {-1, 0, +1} — ternary splits, ternary entropy, feature importance, pruning, and Gini impurity.

## Why This Exists

Standard decision trees split binary thresholds on continuous features. When both your features and labels are inherently ternary — three-level signals, tri-state sensors, approve/abstain/reject decisions — you need a tree that natively handles three-way splits. This crate implements decision trees with ternary branching (less than / equal to / greater than), random forests with bootstrap aggregation and ternary majority voting, plus information-theoretic metrics (ternary entropy, Gini impurity, information gain) designed for the three-class setting. `forbid(unsafe_code)` throughout.

## Core Concepts

- **Ternary splits**: Internal nodes split on `feature < threshold` / `feature == threshold` / `feature > threshold`, producing three children.
- **Ternary entropy**: Shannon entropy over three classes, used for information-gain-based splitting.
- **TernaryDecisionTree**: Build with `fit(samples)`, predict with `predict(features)`. Supports `max_depth` and `min_samples` hyperparameters.
- **Pruning**: Post-hoc pruning against a validation set — converts internal nodes to leaves when it doesn't hurt accuracy.
- **RandomForest**: Ensemble of `TernaryDecisionTree` instances with bootstrap sampling, ternary majority voting, and permutation-based feature importance.
- **Gini impurity**: Complement of squared class probabilities, a purity measure for ternary labels.

## Quick Start

```toml
# Cargo.toml
[dependencies]
ternary-trees = "0.1"
```

```rust
use ternary_trees::{
    Ternary, TernaryDecisionTree, RandomForest,
    Sample, ternary_entropy, gini_impurity, information_gain,
};

fn main() {
    // Training data: features are i8, labels are Ternary
    let samples: Vec<Sample> = vec![
        (vec![-1, -1], Ternary::Neg),
        (vec![-1,  0], Ternary::Neg),
        (vec![ 1,  1], Ternary::Pos),
        (vec![ 1,  0], Ternary::Pos),
        (vec![ 0,  0], Ternary::Zero),
        (vec![ 0,  0], Ternary::Zero),
    ];

    // Decision tree
    let mut tree = TernaryDecisionTree::new(5, 1);
    tree.fit(&samples);
    assert_eq!(tree.predict(&[-1, -1]), Some(Ternary::Neg));
    assert_eq!(tree.predict(&[1, 1]), Some(Ternary::Pos));

    // Random forest with feature importance
    let mut rf = RandomForest::new(5, 5, 1, 2);
    rf.fit(&samples);
    let prediction = rf.predict(&[-1, -1]);
    let importance = rf.feature_importance(&samples, 2);
    println!("Feature importance: {:?}", importance);

    // Information-theoretic metrics
    let h = ternary_entropy([10, 10, 10]); // log₂(3) ≈ 1.585
    let gini = gini_impurity(&samples);
}
```

## API Overview

| Type / Function | Description |
|---|---|
| `Ternary` | Label: `Neg`, `Zero`, `Pos` |
| `Sample` | Type alias: `(Vec<i8>, Ternary)` — features + label |
| `TernaryDecisionTree` | `fit()`, `predict()`, `count_leaves()`, `prune()` |
| `RandomForest` | `fit()`, `predict()`, `accuracy()`, `feature_importance()` |
| `ternary_entropy` | Shannon entropy over 3-class counts |
| `gini_impurity` | Gini impurity for ternary labels |
| `information_gain` | Entropy reduction from a split |
| `count_labels` / `majority` | Label counting and majority vote |

## How It Works

`TernaryDecisionTree::fit` recursively splits the training set. At each node, it evaluates all features at thresholds `-1` and `0`, computing the information gain (parent entropy minus weighted child entropy) of each three-way split. The best split becomes the node's decision rule. Splitting stops when purity reaches 95%, `max_depth` is exceeded, or too few samples remain.

**RandomForest** creates `n_trees` decision trees, each trained on a bootstrap sample (sampling with replacement). Prediction is by majority vote across trees. **Feature importance** permutes each feature and measures the accuracy drop.

**Pruning** is bottom-up: for each internal node, it compares validation accuracy of the subtree vs. a single leaf. If the leaf does no worse, the subtree is replaced.

## Use Cases

- **Tri-state sensor classification**: Classify sensor readings that are naturally {-1, 0, +1} (e.g., magnetometer polarity, ternary logic outputs).
- **Sentiment classification**: Use ternary-labeled features (word sentiment scores) to predict document sentiment.
- **Decision support systems**: Build interpretable models where features and outcomes are approve/abstain/reject.
- **Quality control**: Ternary pass/marginal/fail classification from multi-feature inspections.

## Ecosystem

Part of the **SuperInstance** ternary computing suite:

- `ternary-lattice` — lattice structures for ternary values
- `ternary-codes` — error-correcting codes for ternary data
- `ternary-gradient` — gradient-free optimization on ternary landscapes
- `ternary-language` — ternary NLP and grammar processing
- `ternary-trees` — this crate
- `ternary-transform` — wavelet, Fourier, and kernel transforms
- `ternary-planning` — planning and scheduling with ternary priorities
- `ternary-rl` — reinforcement learning with ternary actions
- `ternary-som` — self-organizing maps for ternary data
- `ternary-failure` — failure analysis with ternary classification

## License

MIT
