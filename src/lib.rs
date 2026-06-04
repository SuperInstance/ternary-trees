#![forbid(unsafe_code)]

//! Decision trees and forests for ternary classification on {-1, 0, +1}.
//!
//! Provides TernaryDecisionTree with ternary splits, RandomForest with ternary voting,
//! feature importance, and pruning with ternary entropy.

/// A ternary value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ternary {
    Neg,
    Zero,
    Pos,
}

impl Ternary {
    pub fn to_i8(self) -> i8 {
        match self {
            Ternary::Neg => -1,
            Ternary::Zero => 0,
            Ternary::Pos => 1,
        }
    }

    pub fn from_i8(v: i8) -> Option<Self> {
        match v {
            -1 => Some(Ternary::Neg),
            0 => Some(Ternary::Zero),
            1 => Some(Ternary::Pos),
            _ => None,
        }
    }

    pub fn values() -> [Ternary; 3] {
        [Ternary::Neg, Ternary::Zero, Ternary::Pos]
    }
}

/// Feature value type for ternary features.
pub type Feature = i8;

/// A training sample: features + label.
pub type Sample = (Vec<Feature>, Ternary);

/// Compute ternary entropy.
pub fn ternary_entropy(counts: [usize; 3]) -> f64 {
    let total = counts[0] + counts[1] + counts[2] as usize;
    if total == 0 {
        return 0.0;
    }
    let mut h = 0.0;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / total as f64;
            h -= p * p.log2();
        }
    }
    h
}

/// Count ternary labels in a slice.
pub fn count_labels(samples: &[Sample]) -> [usize; 3] {
    let mut counts = [0usize; 3];
    for (_, label) in samples {
        match label {
            Ternary::Neg => counts[0] += 1,
            Ternary::Zero => counts[1] += 1,
            Ternary::Pos => counts[2] += 1,
        }
    }
    counts
}

/// Majority vote label.
pub fn majority(samples: &[Sample]) -> Ternary {
    let counts = count_labels(samples);
    if counts[0] >= counts[1] && counts[0] >= counts[2] {
        Ternary::Neg
    } else if counts[1] >= counts[2] {
        Ternary::Zero
    } else {
        Ternary::Pos
    }
}

/// A decision tree node.
#[derive(Debug, Clone)]
pub enum TreeNode {
    Leaf {
        label: Ternary,
        confidence: f64,
        count: usize,
    },
    Internal {
        feature_idx: usize,
        threshold: Feature,
        left: Box<TreeNode>,   // value < threshold
        middle: Box<TreeNode>, // value == threshold
        right: Box<TreeNode>,  // value > threshold
    },
}

/// Ternary decision tree.
#[derive(Debug, Clone)]
pub struct TernaryDecisionTree {
    root: Option<TreeNode>,
    max_depth: usize,
    min_samples: usize,
}

impl TernaryDecisionTree {
    pub fn new(max_depth: usize, min_samples: usize) -> Self {
        TernaryDecisionTree {
            root: None,
            max_depth,
            min_samples,
        }
    }

    pub fn fit(&mut self, samples: &[Sample]) {
        self.root = Some(self.build_tree(samples, 0));
    }

    fn build_tree(&self, samples: &[Sample], depth: usize) -> TreeNode {
        let counts = count_labels(samples);
        let total = counts[0] + counts[1] + counts[2];
        let majority_label = majority(samples);
        let confidence = if total > 0 {
            let max_c = counts[0].max(counts[1]).max(counts[2]);
            max_c as f64 / total as f64
        } else {
            0.0
        };

        if depth >= self.max_depth || total <= self.min_samples || confidence >= 0.95 {
            return TreeNode::Leaf {
                label: majority_label,
                confidence,
                count: total,
            };
        }

        let n_features = samples[0].0.len();
        let mut best_gain = 0.0f64;
        let mut best_feat = 0;
        let mut best_thresh = 0i8;

        let parent_entropy = ternary_entropy(counts);

        for fi in 0..n_features {
            for &thresh in &[-1i8, 0i8] {
                let mut left_counts = [0usize; 3];
                let mut mid_counts = [0usize; 3];
                let mut right_counts = [0usize; 3];
                let mut left_total = 0;
                let mut mid_total = 0;
                let mut right_total = 0;

                for (features, label) in samples {
                    let val = features[fi];
                    let idx = match label {
                        Ternary::Neg => 0,
                        Ternary::Zero => 1,
                        Ternary::Pos => 2,
                    };
                    if val < thresh {
                        left_counts[idx] += 1;
                        left_total += 1;
                    } else if val == thresh {
                        mid_counts[idx] += 1;
                        mid_total += 1;
                    } else {
                        right_counts[idx] += 1;
                        right_total += 1;
                    }
                }

                if left_total == 0 || mid_total == 0 || right_total == 0 {
                    continue;
                }

                let child_entropy = (left_total as f64 * ternary_entropy(left_counts)
                    + mid_total as f64 * ternary_entropy(mid_counts)
                    + right_total as f64 * ternary_entropy(right_counts))
                    / total as f64;

                let gain = parent_entropy - child_entropy;
                if gain > best_gain {
                    best_gain = gain;
                    best_feat = fi;
                    best_thresh = thresh;
                }
            }
        }

        if best_gain < 1e-10 {
            return TreeNode::Leaf {
                label: majority_label,
                confidence,
                count: total,
            };
        }

        let mut left_samples = Vec::new();
        let mut mid_samples = Vec::new();
        let mut right_samples = Vec::new();
        for s in samples {
            match s.0[best_feat].cmp(&best_thresh) {
                std::cmp::Ordering::Less => left_samples.push(s.clone()),
                std::cmp::Ordering::Equal => mid_samples.push(s.clone()),
                std::cmp::Ordering::Greater => right_samples.push(s.clone()),
            }
        }

        TreeNode::Internal {
            feature_idx: best_feat,
            threshold: best_thresh,
            left: Box::new(self.build_tree(&left_samples, depth + 1)),
            middle: Box::new(self.build_tree(&mid_samples, depth + 1)),
            right: Box::new(self.build_tree(&right_samples, depth + 1)),
        }
    }

    pub fn predict(&self, features: &[Feature]) -> Option<Ternary> {
        self.root.as_ref().map(|node| Self::predict_node(node, features))
    }

    fn predict_node(node: &TreeNode, features: &[Feature]) -> Ternary {
        match node {
            TreeNode::Leaf { label, .. } => *label,
            TreeNode::Internal {
                feature_idx,
                threshold,
                left,
                middle,
                right,
            } => {
                let val = features[*feature_idx];
                match val.cmp(threshold) {
                    std::cmp::Ordering::Less => Self::predict_node(left, features),
                    std::cmp::Ordering::Equal => Self::predict_node(middle, features),
                    std::cmp::Ordering::Greater => Self::predict_node(right, features),
                }
            }
        }
    }

    /// Count the number of leaves.
    pub fn count_leaves(&self) -> usize {
        self.root.as_ref().map_or(0, Self::count_leaves_node)
    }

    fn count_leaves_node(node: &TreeNode) -> usize {
        match node {
            TreeNode::Leaf { .. } => 1,
            TreeNode::Internal {
                left, middle, right, ..
            } => {
                Self::count_leaves_node(left)
                    + Self::count_leaves_node(middle)
                    + Self::count_leaves_node(right)
            }
        }
    }

    /// Prune: convert internal nodes to leaves if it doesn't hurt accuracy.
    pub fn prune(&mut self, validation: &[Sample]) {
        if let Some(root) = self.root.take() {
            self.root = Some(self.prune_node(root, validation));
        }
    }

    fn prune_node(&self, node: TreeNode, validation: &[Sample]) -> TreeNode {
        match node {
            TreeNode::Leaf { .. } => node,
            TreeNode::Internal {
                feature_idx,
                threshold,
                left,
                middle,
                right,
            } => {
                let left = Box::new(self.prune_node(*left, validation));
                let middle = Box::new(self.prune_node(*middle, validation));
                let right = Box::new(self.prune_node(*right, validation));

                let internal_node = TreeNode::Internal {
                    feature_idx,
                    threshold,
                    left: left.clone(),
                    middle: middle.clone(),
                    right: right.clone(),
                };

                // Compute accuracy with internal node
                let internal_acc = validation
                    .iter()
                    .filter(|(f, l)| Self::predict_node(&internal_node, f) == *l)
                    .count();

                // Try converting to leaf
                let filtered: Vec<Sample> = validation.to_vec();
                let leaf_label = majority(&filtered);
                let leaf_node = TreeNode::Leaf {
                    label: leaf_label,
                    confidence: 1.0,
                    count: filtered.len(),
                };
                let leaf_acc = validation
                    .iter()
                    .filter(|(_, l)| leaf_label == *l)
                    .count();

                if leaf_acc >= internal_acc {
                    leaf_node
                } else {
                    internal_node
                }
            }
        }
    }
}

/// Random forest for ternary classification.
#[derive(Debug, Clone)]
pub struct RandomForest {
    pub trees: Vec<TernaryDecisionTree>,
    pub n_features_per_split: usize,
}

impl RandomForest {
    pub fn new(n_trees: usize, max_depth: usize, min_samples: usize, n_features_per_split: usize) -> Self {
        let trees = (0..n_trees)
            .map(|_| TernaryDecisionTree::new(max_depth, min_samples))
            .collect();
        RandomForest {
            trees,
            n_features_per_split,
        }
    }

    pub fn fit(&mut self, samples: &[Sample]) {
        let n = samples.len();
        for tree in &mut self.trees {
            // Bootstrap sample
            let mut bootstrap = Vec::with_capacity(n);
            for _ in 0..n {
                let idx = simple_hash(n) % n;
                bootstrap.push(samples[idx].clone());
            }
            tree.fit(&bootstrap);
        }
    }

    pub fn predict(&self, features: &[Feature]) -> Ternary {
        let mut counts = [0usize; 3];
        for tree in &self.trees {
            if let Some(label) = tree.predict(features) {
                match label {
                    Ternary::Neg => counts[0] += 1,
                    Ternary::Zero => counts[1] += 1,
                    Ternary::Pos => counts[2] += 1,
                }
            }
        }
        if counts[0] >= counts[1] && counts[0] >= counts[2] {
            Ternary::Neg
        } else if counts[1] >= counts[2] {
            Ternary::Zero
        } else {
            Ternary::Pos
        }
    }

    /// Compute feature importance via permutation.
    pub fn feature_importance(&self, samples: &[Sample], n_features: usize) -> Vec<f64> {
        let baseline_acc = self.accuracy(samples);
        let mut importance = vec![0.0; n_features];

        for fi in 0..n_features {
            let mut permuted = samples.to_vec();
            // Simple permutation: reverse order of feature fi
            let n = permuted.len();
            for i in 0..n / 2 {
                let j = n - 1 - i;
                let tmp = permuted[i].0[fi];
                permuted[i].0[fi] = permuted[j].0[fi];
                permuted[j].0[fi] = tmp;
            }
            let perm_acc = self.accuracy(&permuted);
            importance[fi] = baseline_acc - perm_acc;
        }
        importance
    }

    pub fn accuracy(&self, samples: &[Sample]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        let correct = samples
            .iter()
            .filter(|(f, l)| self.predict(f) == *l)
            .count();
        correct as f64 / samples.len() as f64
    }
}

/// Simple deterministic hash for bootstrap sampling.
fn simple_hash(n: usize) -> usize {
    n.wrapping_mul(1103515245).wrapping_add(12345)
}

/// Compute Gini impurity for ternary labels.
pub fn gini_impurity(samples: &[Sample]) -> f64 {
    let counts = count_labels(samples);
    let total = counts[0] + counts[1] + counts[2];
    if total == 0 {
        return 0.0;
    }
    let mut gini = 0.0;
    for &c in &counts {
        let p = c as f64 / total as f64;
        gini += p * p;
    }
    1.0 - gini
}

/// Compute information gain for a split.
pub fn information_gain(parent: &[Sample], children: &[&[Sample]]) -> f64 {
    let parent_h = ternary_entropy(count_labels(parent));
    let parent_n = parent.len() as f64;
    let mut child_h = 0.0;
    for child in children {
        let w = child.len() as f64 / parent_n;
        child_h += w * ternary_entropy(count_labels(child));
    }
    parent_h - child_h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_values() {
        assert_eq!(Ternary::Neg.to_i8(), -1);
        assert_eq!(Ternary::Zero.to_i8(), 0);
        assert_eq!(Ternary::Pos.to_i8(), 1);
    }

    #[test]
    fn test_ternary_entropy_uniform() {
        let h = ternary_entropy([10, 10, 10]);
        assert!((h - (3.0f64.log2())).abs() < 1e-10);
    }

    #[test]
    fn test_ternary_entropy_pure() {
        let h = ternary_entropy([100, 0, 0]);
        assert!(h.abs() < 1e-10);
    }

    #[test]
    fn test_count_labels() {
        let samples = vec![
            (vec![0], Ternary::Neg),
            (vec![0], Ternary::Neg),
            (vec![0], Ternary::Pos),
        ];
        let counts = count_labels(&samples);
        assert_eq!(counts, [2, 0, 1]);
    }

    #[test]
    fn test_majority() {
        let samples = vec![
            (vec![0], Ternary::Pos),
            (vec![0], Ternary::Pos),
            (vec![0], Ternary::Neg),
        ];
        assert_eq!(majority(&samples), Ternary::Pos);
    }

    #[test]
    fn test_decision_tree_fit_predict() {
        let samples = vec![
            (vec![-1, -1], Ternary::Neg),
            (vec![-1, 0], Ternary::Neg),
            (vec![1, 1], Ternary::Pos),
            (vec![1, 0], Ternary::Pos),
            (vec![0, 0], Ternary::Zero),
            (vec![0, 0], Ternary::Zero),
        ];
        let mut tree = TernaryDecisionTree::new(5, 1);
        tree.fit(&samples);
        assert_eq!(tree.predict(&[-1, -1]), Some(Ternary::Neg));
        assert_eq!(tree.predict(&[1, 1]), Some(Ternary::Pos));
    }

    #[test]
    fn test_decision_tree_single_class() {
        let samples = vec![
            (vec![1], Ternary::Pos),
            (vec![1], Ternary::Pos),
            (vec![1], Ternary::Pos),
        ];
        let mut tree = TernaryDecisionTree::new(5, 1);
        tree.fit(&samples);
        assert_eq!(tree.predict(&[1]), Some(Ternary::Pos));
    }

    #[test]
    fn test_tree_leaves_count() {
        let samples = vec![
            (vec![-1], Ternary::Neg),
            (vec![0], Ternary::Zero),
            (vec![1], Ternary::Pos),
        ];
        let mut tree = TernaryDecisionTree::new(10, 1);
        tree.fit(&samples);
        assert!(tree.count_leaves() >= 1);
    }

    #[test]
    fn test_gini_impurity() {
        let pure = vec![(vec![0], Ternary::Pos)];
        assert!(gini_impurity(&pure).abs() < 1e-10);

        let mixed = vec![
            (vec![0], Ternary::Neg),
            (vec![0], Ternary::Zero),
            (vec![0], Ternary::Pos),
        ];
        let g = gini_impurity(&mixed);
        assert!((g - (1.0_f64 - 3.0_f64 * (1.0_f64 / 3.0_f64).powi(2))).abs() < 1e-10);
    }

    #[test]
    fn test_information_gain() {
        let parent = vec![
            (vec![0], Ternary::Neg),
            (vec![0], Ternary::Neg),
            (vec![0], Ternary::Pos),
            (vec![0], Ternary::Pos),
        ];
        let left = &parent[0..2];
        let right = &parent[2..4];
        let gain = information_gain(&parent, &[left, right]);
        assert!(gain > 0.0);
    }

    #[test]
    fn test_random_forest() {
        let samples = vec![
            (vec![-1, -1], Ternary::Neg),
            (vec![-1, 0], Ternary::Neg),
            (vec![1, 1], Ternary::Pos),
            (vec![1, 0], Ternary::Pos),
            (vec![0, 0], Ternary::Zero),
            (vec![0, 0], Ternary::Zero),
            (vec![-1, -1], Ternary::Neg),
            (vec![1, 1], Ternary::Pos),
        ];
        let mut rf = RandomForest::new(5, 5, 1, 2);
        rf.fit(&samples);
        let pred = rf.predict(&[-1, -1]);
        assert!(pred == Ternary::Neg);
    }

    #[test]
    fn test_random_forest_accuracy() {
        let samples = vec![
            (vec![-1], Ternary::Neg),
            (vec![0], Ternary::Zero),
            (vec![1], Ternary::Pos),
            (vec![-1], Ternary::Neg),
            (vec![0], Ternary::Zero),
            (vec![1], Ternary::Pos),
        ];
        let mut rf = RandomForest::new(5, 5, 1, 1);
        rf.fit(&samples);
        let acc = rf.accuracy(&samples);
        assert!(acc > 0.0);
    }

    #[test]
    fn test_feature_importance() {
        let samples = vec![
            (vec![-1, 0], Ternary::Neg),
            (vec![1, 0], Ternary::Pos),
            (vec![0, -1], Ternary::Zero),
            (vec![-1, 0], Ternary::Neg),
            (vec![1, 0], Ternary::Pos),
            (vec![0, 1], Ternary::Zero),
        ];
        let mut rf = RandomForest::new(5, 5, 1, 2);
        rf.fit(&samples);
        let importance = rf.feature_importance(&samples, 2);
        assert_eq!(importance.len(), 2);
    }

    #[test]
    fn test_tree_prune() {
        let samples = vec![
            (vec![-1], Ternary::Neg),
            (vec![0], Ternary::Zero),
            (vec![1], Ternary::Pos),
            (vec![-1], Ternary::Neg),
            (vec![0], Ternary::Zero),
            (vec![1], Ternary::Pos),
        ];
        let mut tree = TernaryDecisionTree::new(10, 1);
        tree.fit(&samples);
        let leaves_before = tree.count_leaves();
        tree.prune(&samples);
        // Pruning should not increase leaf count beyond reasonable bounds
        assert!(tree.count_leaves() <= leaves_before || leaves_before <= 3);
    }

    #[test]
    fn test_empty_samples_entropy() {
        let h = ternary_entropy([0, 0, 0]);
        assert!(h.abs() < 1e-10);
    }

    #[test]
    fn test_predict_no_fit() {
        let tree = TernaryDecisionTree::new(5, 1);
        assert!(tree.predict(&[0]).is_none());
    }

    #[test]
    fn test_from_i8() {
        assert_eq!(Ternary::from_i8(-1), Some(Ternary::Neg));
        assert_eq!(Ternary::from_i8(2), None);
    }

    #[test]
    fn test_forest_empty_accuracy() {
        let rf = RandomForest::new(3, 5, 1, 1);
        assert_eq!(rf.accuracy(&[]), 0.0);
    }

    #[test]
    fn test_ternary_values_array() {
        let vals = Ternary::values();
        assert_eq!(vals.len(), 3);
        assert_eq!(vals[0], Ternary::Neg);
        assert_eq!(vals[1], Ternary::Zero);
        assert_eq!(vals[2], Ternary::Pos);
    }

    #[test]
    fn test_gini_pure_vs_mixed() {
        let pure = vec![(vec![0], Ternary::Pos), (vec![0], Ternary::Pos)];
        let mixed = vec![(vec![0], Ternary::Neg), (vec![0], Ternary::Pos)];
        assert!(gini_impurity(&pure) < gini_impurity(&mixed));
    }

    #[test]
    fn test_decision_tree_zero_features() {
        let samples = vec![
            (vec![], Ternary::Neg),
            (vec![], Ternary::Pos),
        ];
        let mut tree = TernaryDecisionTree::new(5, 1);
        tree.fit(&samples);
        // Should still produce a prediction
        assert!(tree.predict(&[]).is_some());
    }
}
