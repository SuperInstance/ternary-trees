# Future Integration: ternary-trees

## Current State
Implements decision trees for ternary classification: `TernaryDecisionTree` with three-way splits (left/middle/right branches matching Neg/Zero/Pos), `RandomForest` with ternary voting, feature importance scoring, and ternary entropy-based pruning.

## Integration Opportunities

### With ternary-cell / room-as-codespace
A decision tree classifies room situations. Each feature is a room parameter (temperature trit, occupancy trit, energy trit). The tree's `predict()` outputs a ternary classification: the room needs cooling (Neg), is fine (Zero), or needs heating (Pos). The `three-way split` naturally matches ternary features — no binarization needed. `RandomForest` aggregates classifications from multiple trees for robustness.

### With ternary-attention
Use feature importance from `RandomForest::feature_importance()` to weight attention heads. Features with high importance get more attention weight. The forest learns *which room parameters matter most* and the attention mechanism focuses computation on those parameters. This creates an attention system that adapts to each room's specific characteristics.

### With ternary-clustering
Decision tree leaf nodes define natural clusters. Each leaf represents a set of rooms with similar ternary feature profiles. The tree provides interpretable clustering — you can trace the decision path to understand *why* rooms are grouped together, unlike black-box k-means.

## Potential in Mature Systems
In PLATO, each construct maintains a small `TernaryDecisionTree` as its decision-making skill. The tree is trained on historical outcomes: given room state X, action Y led to outcome Z. At runtime, `predict()` provides fast O(depth) decisions. The tree is interpretable — construct provenance chains can explain *why* a decision was made by tracing the split sequence. On ESP32, the tree compiles to a series of if-else comparisons — zero memory allocation.

## Cross-Pollination Ideas
**Music × Trees:** Classify chord quality with ternary features. Input features = interval trits (down/unison/up from root). The tree classifies: major (Pos), minor (Neg), suspended (Zero). `RandomForest` over multiple music theory features (scale degree, voice leading, rhythmic position) produces robust harmonic analysis. Connects to `ternary-music`.

**Game theory × Trees:** Extensive-form games ARE trees. Each node is a decision point, branches are ternary actions (cooperate/defect/wait). Minimax on ternary trees finds Nash equilibria in sequential games. `ternary-game-theory` benefits from the tree structure.

## Dependencies for Next Steps
- Online learning: update trees incrementally as room data streams in
- Tree depth limits for ESP32 deployment (max 8 levels = 256 leaves)
- Integration with `ternary-cell` as a decision skill trait
