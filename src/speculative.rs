use crate::radix_cache::{CacheLookup, TokenId};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NextTokenPrediction {
    pub token: TokenId,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerifyStep {
    pub expected: TokenId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeVerifyResult {
    pub expected: Vec<TokenId>, // Target model's expected token for each node in the tree
}

/// A tree of drafted tokens for parallel verification.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeculativeTree {
    pub nodes: Vec<TreeNode>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TreeNode {
    pub token: TokenId,
    pub parent: Option<usize>, // Index of parent node in SpeculativeTree::nodes
}

impl SpeculativeTree {
    pub fn new(token: TokenId) -> Self {
        Self {
            nodes: vec![TreeNode { token, parent: None }],
        }
    }

    pub fn add_child(&mut self, parent_idx: usize, token: TokenId) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(TreeNode {
            token,
            parent: Some(parent_idx),
        });
        idx
    }

    /// Returns the sequence of tokens from the root to the given node.
    pub fn get_path(&self, mut node_idx: usize) -> Vec<TokenId> {
        let mut path = Vec::new();
        loop {
            let node = &self.nodes[node_idx];
            path.push(node.token);
            if let Some(parent) = node.parent {
                node_idx = parent;
            } else {
                break;
            }
        }
        path.reverse();
        path
    }

    pub fn find_best_path(&self, verified: &TreeVerifyResult) -> (Vec<TokenId>, Option<TokenId>) {
        let mut best_path = Vec::new();
        let mut rejected_token = None;
        let mut max_len = 0;

        let mut valid = vec![false; self.nodes.len()];
        valid[0] = true;

        for i in 1..self.nodes.len() {
            if let Some(parent) = self.nodes[i].parent {
                if valid[parent] && self.nodes[i].token == verified.expected[parent] {
                    valid[i] = true;
                }
            }
        }

        for i in 0..self.nodes.len() {
            if valid[i] {
                let path = self.get_path(i);
                if path.len() >= max_len {
                    max_len = path.len();
                    best_path = path;
                    rejected_token = if i < verified.expected.len() {
                        Some(verified.expected[i])
                    } else {
                        None
                    };
                }
            }
        }

        (best_path, rejected_token)
    }
}

pub trait DraftModel {
    fn bind_prefix_cache(&mut self, _prefix: &CacheLookup) -> Result<()> {
        Ok(())
    }

    fn bind_slot(&mut self, _slot: crate::slot_manager::SlotId) -> Result<()> {
        Ok(())
    }

    fn switch_model(&mut self, _name: &str, _pool: &crate::model_pool::ModelPool) -> Result<()> {
        Ok(())
    }

    fn draft(
        &mut self,
        context: &[TokenId],
        max_tokens: usize,
        matcher: Option<&mut (dyn crate::constraints::CfgMatcher + '_)>,
    ) -> Result<Vec<NextTokenPrediction>>;

    fn draft_tree(
        &mut self,
        context: &[TokenId],
        _max_tokens: usize,
        _width: usize,
        _matcher: Option<&mut (dyn crate::constraints::CfgMatcher + '_)>,
    ) -> Result<SpeculativeTree> {
        // Default implementation: just a linear chain
        let linear = self.draft(context, _max_tokens, _matcher)?;
        if linear.is_empty() {
            return Err(SpeculativeError::Model("Draft model returned no tokens".into()));
        }
        let mut tree = SpeculativeTree::new(linear[0].token);
        let mut last_idx = 0;
        for pred in linear.into_iter().skip(1) {
            last_idx = tree.add_child(last_idx, pred.token);
        }
        Ok(tree)
    }

    fn draft_batch(
        &mut self,
        requests: &mut [(&[TokenId], usize, Option<&mut (dyn crate::constraints::CfgMatcher + '_)>)],
    ) -> Result<Vec<Vec<NextTokenPrediction>>> {
        let mut results = Vec::with_capacity(requests.len());
        for (ctx, max, matcher) in requests {
            results.push(self.draft(ctx, *max, matcher.as_mut().map(|m| *m as &mut (dyn crate::constraints::CfgMatcher + '_)))?);
        }
        Ok(results)
    }
}

pub trait TargetModel {
    fn bind_prefix_cache(&mut self, _prefix: &CacheLookup) -> Result<()> {
        Ok(())
    }

    fn bind_slot(&mut self, _slot: crate::slot_manager::SlotId) -> Result<()> {
        Ok(())
    }

    fn switch_model(&mut self, _name: &str, _pool: &crate::model_pool::ModelPool) -> Result<()> {
        Ok(())
    }

    fn verify(
        &mut self,
        context: &[TokenId],
        drafted: &[TokenId],
        matcher: Option<&mut (dyn crate::constraints::CfgMatcher + '_)>,
    ) -> Result<Vec<VerifyStep>>;

    fn verify_tree(
        &mut self,
        context: &[TokenId],
        tree: &SpeculativeTree,
        _matcher: Option<&mut (dyn crate::constraints::CfgMatcher + '_)>,
    ) -> Result<TreeVerifyResult> {
        // Default implementation: verify the first path (leftmost branch)
        let mut path_indices = Vec::new();
        let mut curr = 0;
        loop {
            path_indices.push(curr);
            if let Some(first_child) = tree.nodes.iter().position(|n| n.parent == Some(curr)) {
                curr = first_child;
            } else {
                break;
            }
        }
        let path: Vec<TokenId> = path_indices.iter().map(|&idx| tree.nodes[idx].token).collect();
        let verified = self.verify(context, &path, _matcher)?;
        
        let mut expected = vec![0; tree.nodes.len()]; // 0 as placeholder
        for (i, step) in path_indices.into_iter().zip(verified) {
            expected[i] = step.expected;
        }
        Ok(TreeVerifyResult { expected })
    }

    fn verify_batch(
        &mut self,
        requests: &mut [(&[TokenId], &[TokenId], Option<&mut (dyn crate::constraints::CfgMatcher + '_)>)],
    ) -> Result<Vec<Vec<VerifyStep>>> {
        let mut results = Vec::with_capacity(requests.len());
        for (ctx, drafted, matcher) in requests {
            results.push(self.verify(ctx, drafted, matcher.as_mut().map(|m| *m as &mut (dyn crate::constraints::CfgMatcher + '_)))?);
        }
        Ok(results)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeDecoder {
    draft_window: usize,
    max_window: usize,
}

impl SpeculativeDecoder {
    pub fn draft_window(&self) -> usize {
        self.draft_window
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeSession {
    draft_window: usize,
    max_window: usize,
    prompt: Vec<TokenId>,
    context: Vec<TokenId>,
    stats: SpeculativeStats,
    rejected_token: Option<TokenId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeOutput {
    pub tokens: Vec<TokenId>,
    pub stats: SpeculativeStats,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpeculativeStats {
    pub draft_calls: usize,
    pub target_calls: usize,
    pub drafted_tokens: usize,
    pub accepted_tokens: usize,
    pub rejected_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpeculativeError {
    EmptyDraftWindow,
    DraftReturnedTooMany {
        requested: usize,
        returned: usize,
    },
    TargetReturnedWrongLength {
        drafted: usize,
        returned: usize,
    },
    Model(String),
}

pub type Result<T> = std::result::Result<T, SpeculativeError>;

impl SpeculativeDecoder {
    pub fn new(draft_window: usize) -> Result<Self> {
        if draft_window == 0 {
            return Err(SpeculativeError::EmptyDraftWindow);
        }

        Ok(Self { draft_window, max_window: draft_window * 2 })
    }

    pub fn with_max_window(draft_window: usize, max_window: usize) -> Result<Self> {
        if draft_window == 0 || max_window < draft_window {
            return Err(SpeculativeError::EmptyDraftWindow);
        }
        Ok(Self { draft_window, max_window })
    }

    pub fn generate<D, T>(
        &self,
        draft_model: &mut D,
        target_model: &mut T,
        prompt: &[TokenId],
        max_new_tokens: usize,
    ) -> Result<SpeculativeOutput>
    where
        D: DraftModel,
        T: TargetModel,
    {
        let mut session = self.begin(prompt)?;
        let mut generated = Vec::with_capacity(max_new_tokens);

        while generated.len() < max_new_tokens {
            let remaining = max_new_tokens - generated.len();
            let drafted = session.draft(draft_model, target_model, remaining)?;

            if drafted.is_empty() && !session.has_pending_rejection() {
                break;
            }

            generated.extend_from_slice(&drafted);
            if !session.has_pending_rejection() {
                continue;
            }

            if let Some(token) = session.take_rejected_token() {
                generated.push(token);
            }
        }

        Ok(SpeculativeOutput {
            tokens: generated,
            stats: session.stats().clone(),
        })
    }

    pub fn begin(&self, prompt: &[TokenId]) -> Result<SpeculativeSession> {
        Ok(SpeculativeSession {
            draft_window: self.draft_window,
            max_window: self.max_window,
            prompt: prompt.to_vec(),
            context: prompt.to_vec(),
            stats: SpeculativeStats::default(),
            rejected_token: None,
        })
    }
}

impl SpeculativeSession {
    pub fn prompt(&self) -> &[TokenId] {
        &self.prompt
    }

    pub fn context(&self) -> &[TokenId] {
        &self.context
    }

    pub fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }

    pub fn has_pending_rejection(&self) -> bool {
        self.rejected_token.is_some()
    }

    pub fn take_rejected_token(&mut self) -> Option<TokenId> {
        self.rejected_token.take()
    }

    pub fn current_window(&self) -> usize {
        self.draft_window
    }

    pub fn draft<D, T>(
        &mut self,
        draft_model: &mut D,
        target_model: &mut T,
        max_new_tokens: usize,
    ) -> Result<Vec<TokenId>>
    where
        D: DraftModel,
        T: TargetModel,
    {
        let requested = max_new_tokens.min(self.draft_window);
        let predictions = draft_model.draft(&self.context, requested, None)?;
        self.stats.draft_calls += 1;

        if predictions.len() > requested {
            return Err(SpeculativeError::DraftReturnedTooMany {
                requested,
                returned: predictions.len(),
            });
        }

        if predictions.is_empty() {
            return Ok(Vec::new());
        }

        let drafted = predictions
            .iter()
            .map(|prediction| prediction.token)
            .collect::<Vec<_>>();
        self.stats.drafted_tokens += drafted.len();

        let verified = target_model.verify(&self.context, &drafted, None)?;
        self.stats.target_calls += 1;

        if verified.len() != drafted.len() {
            return Err(SpeculativeError::TargetReturnedWrongLength {
                drafted: drafted.len(),
                returned: verified.len(),
            });
        }

        let mut accepted_this_round = 0;
        let mut rejected = None;

        for (index, (draft_token, target_step)) in drafted.iter().zip(&verified).enumerate() {
            if *draft_token == target_step.expected {
                accepted_this_round += 1;
                continue;
            }

            rejected = Some((index, target_step.expected));
            break;
        }

        let mut accepted = Vec::with_capacity(accepted_this_round);
        if accepted_this_round > 0 {
            accepted.extend_from_slice(&drafted[..accepted_this_round]);
            self.context.extend_from_slice(&accepted);
            self.stats.accepted_tokens += accepted_this_round;
        }

        if let Some((rejected_index, target_token)) = rejected {
            self.context.push(target_token);
            self.rejected_token = Some(target_token);
            self.stats.rejected_tokens += drafted.len() - rejected_index;
        } else {
            self.rejected_token = None;
        }

        self.adjust_window(accepted_this_round, drafted.len());
        Ok(accepted)
    }

    pub fn commit(
        &mut self,
        drafted: &[NextTokenPrediction],
        verified: &[VerifyStep],
    ) -> Result<Vec<TokenId>> {
        if verified.len() != drafted.len() {
            return Err(SpeculativeError::TargetReturnedWrongLength {
                drafted: drafted.len(),
                returned: verified.len(),
            });
        }

        self.stats.drafted_tokens += drafted.len();

        let mut accepted_this_round = 0;
        let mut rejected = None;

        for (index, (draft_pred, target_step)) in drafted.iter().zip(verified).enumerate() {
            if draft_pred.token == target_step.expected {
                accepted_this_round += 1;
                continue;
            }

            rejected = Some((index, target_step.expected));
            break;
        }

        let mut accepted = Vec::with_capacity(accepted_this_round);
        if accepted_this_round > 0 {
            let tokens: Vec<TokenId> =
                drafted[..accepted_this_round].iter().map(|p| p.token).collect();
            accepted.extend_from_slice(&tokens);
            self.context.extend_from_slice(&tokens);
            self.stats.accepted_tokens += accepted_this_round;
        }

        if let Some((rejected_index, target_token)) = rejected {
            self.context.push(target_token);
            self.rejected_token = Some(target_token);
            self.stats.rejected_tokens += drafted.len() - rejected_index;
        } else {
            self.rejected_token = None;
        }

        self.adjust_window(accepted_this_round, drafted.len());
        Ok(accepted)
    }

    pub fn record_draft_call(&mut self) {
        self.stats.draft_calls += 1;
    }

    pub fn record_target_call(&mut self) {
        self.stats.target_calls += 1;
    }

    fn adjust_window(&mut self, accepted_count: usize, drafted_count: usize) {
        if accepted_count == drafted_count {
            if self.draft_window < self.max_window {
                self.draft_window += 1;
            }
        } else {
            self.draft_window = (accepted_count + 1).max(1);
        }
    }
}

impl std::fmt::Display for SpeculativeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyDraftWindow => write!(formatter, "draft window must be greater than zero"),
            Self::DraftReturnedTooMany {
                requested,
                returned,
            } => write!(
                formatter,
                "draft model returned {returned} tokens after {requested} were requested"
            ),
            Self::TargetReturnedWrongLength { drafted, returned } => write!(
                formatter,
                "target model returned {returned} verification steps for {drafted} drafted tokens"
            ),
            Self::Model(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for SpeculativeError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct ScriptedDraft {
        script: Vec<TokenId>,
    }

    impl DraftModel for ScriptedDraft {
        fn draft(
            &mut self,
            context: &[TokenId],
            max_tokens: usize,
            _matcher: Option<&mut (dyn crate::constraints::CfgMatcher + '_)>,
        ) -> Result<Vec<NextTokenPrediction>> {
            Ok(self.script[context.len()..]
                .iter()
                .take(max_tokens)
                .map(|token| NextTokenPrediction {
                    token: *token,
                    confidence: 1.0,
                })
                .collect())
        }
    }

    #[derive(Debug)]
    struct ScriptedTarget {
        script: Vec<TokenId>,
    }

    impl TargetModel for ScriptedTarget {
        fn verify(&mut self, context: &[TokenId], drafted: &[TokenId], _matcher: Option<&mut (dyn crate::constraints::CfgMatcher + '_)>) -> Result<Vec<VerifyStep>> {
            Ok(self.script[context.len()..]
                .iter()
                .take(drafted.len())
                .map(|token| VerifyStep { expected: *token })
                .collect())
        }
    }

    #[test]
    fn accepts_full_draft_windows() {
        let decoder = SpeculativeDecoder::new(3).unwrap();
        let prompt = [1, 2];
        let mut draft = ScriptedDraft {
            script: vec![1, 2, 3, 4, 5, 6],
        };
        let mut target = ScriptedTarget {
            script: vec![1, 2, 3, 4, 5, 6],
        };

        let output = decoder
            .generate(&mut draft, &mut target, &prompt, 4)
            .unwrap();

        assert_eq!(output.tokens, vec![3, 4, 5, 6]);
        assert_eq!(
            output.stats,
            SpeculativeStats {
                draft_calls: 2,
                target_calls: 2,
                drafted_tokens: 4,
                accepted_tokens: 4,
                rejected_tokens: 0,
            }
        );
    }

    #[test]
    fn falls_back_to_target_token_on_first_rejection() {
        let decoder = SpeculativeDecoder::new(4).unwrap();
        let prompt = [1, 2];
        let mut draft = ScriptedDraft {
            script: vec![1, 2, 9, 4, 5],
        };
        let mut target = ScriptedTarget {
            script: vec![1, 2, 3, 4, 5],
        };

        let output = decoder
            .generate(&mut draft, &mut target, &prompt, 3)
            .unwrap();

        assert_eq!(output.tokens, vec![3, 4, 5]);
        assert_eq!(output.stats.accepted_tokens, 2);
        assert_eq!(output.stats.rejected_tokens, 3);
    }

    #[test]
    fn stops_at_max_new_tokens() {
        let decoder = SpeculativeDecoder::new(8).unwrap();
        let prompt = [1];
        let mut draft = ScriptedDraft {
            script: vec![1, 2, 3, 4, 5],
        };
        let mut target = ScriptedTarget {
            script: vec![1, 2, 3, 4, 5],
        };

        let output = decoder
            .generate(&mut draft, &mut target, &prompt, 2)
            .unwrap();

        assert_eq!(output.tokens, vec![2, 3]);
        assert_eq!(output.stats.drafted_tokens, 2);
    }

    #[test]
    fn rejects_zero_draft_window() {
        assert_eq!(
            SpeculativeDecoder::new(0),
            Err(SpeculativeError::EmptyDraftWindow)
        );
    }

    #[test]
    fn speculative_error_display() {
        assert!(format!("{}", SpeculativeError::EmptyDraftWindow).contains("draft window must be greater than zero"));
        assert!(format!("{}", SpeculativeError::DraftReturnedTooMany { requested: 1, returned: 2 }).contains("draft model returned 2 tokens"));
        assert!(format!("{}", SpeculativeError::TargetReturnedWrongLength { drafted: 1, returned: 2 }).contains("target model returned 2 verification steps"));
        assert!(format!("{}", SpeculativeError::Model("oops".into())).contains("oops"));
    }

    #[test]
    fn speculative_stats_ops() {
        let s1 = SpeculativeStats { draft_calls: 1, ..Default::default() };
        let s2 = SpeculativeStats { draft_calls: 2, ..Default::default() };
        assert_eq!(s1.draft_calls + s2.draft_calls, 3);
    }

    #[test]
    fn session_exposes_prompt_and_context() {
        let decoder = SpeculativeDecoder::new(2).unwrap();
        let session = decoder.begin(&[7, 8]).unwrap();

        assert_eq!(session.prompt(), &[7, 8]);
        assert_eq!(session.context(), &[7, 8]);
    }

    #[test]
    fn batch_default_impls() {
        struct D; impl DraftModel for D { fn draft(&mut self, _: &[TokenId], m: usize, _: Option<&mut (dyn crate::constraints::CfgMatcher + '_)>) -> Result<Vec<NextTokenPrediction>> { Ok(vec![NextTokenPrediction{token:1, confidence:1.0}; m]) } }
        struct T; impl TargetModel for T { fn verify(&mut self, _: &[TokenId], d: &[TokenId], _: Option<&mut (dyn crate::constraints::CfgMatcher + '_)>) -> Result<Vec<VerifyStep>> { Ok(vec![VerifyStep{expected:1}; d.len()]) } }
       
        let mut d = D;
        let mut t = T;
        assert!(d.draft_batch(&mut [(&[1], 1, None)]).is_ok());
        assert!(t.verify_batch(&mut [(&[1], &[1], None)]).is_ok());
        assert!(d.bind_slot(crate::slot_manager::SlotId(0)).is_ok());
        assert!(t.bind_slot(crate::slot_manager::SlotId(0)).is_ok());
    }

    #[test]
    fn commit_logic() {
        let decoder = SpeculativeDecoder::new(4).unwrap();
        let mut session = decoder.begin(&[1]).unwrap();
        let drafted = vec![NextTokenPrediction{token: 2, confidence: 1.0}];
        let verified = vec![VerifyStep{expected: 2}];
        let accepted = session.commit(&drafted, &verified).unwrap();
        assert_eq!(accepted, vec![2]);
        assert_eq!(session.context(), vec![1, 2]);
       
        // Error path
        assert!(session.commit(&drafted, &[]).is_err());
    }

    #[test]
    fn record_stats() {
        let decoder = SpeculativeDecoder::new(4).unwrap();
        let mut session = decoder.begin(&[]).unwrap();
        session.record_draft_call();
        session.record_target_call();
        assert_eq!(session.stats().draft_calls, 1);
        assert_eq!(session.stats().target_calls, 1);
    }

    #[test]
    fn draft_error_paths() {
        let decoder = SpeculativeDecoder::new(4).unwrap();
        let mut session = decoder.begin(&[1]).unwrap();
       
        struct BadDraft;
        impl DraftModel for BadDraft {
            fn draft(&mut self, _: &[TokenId], _: usize, _: Option<&mut (dyn crate::constraints::CfgMatcher + '_)>) -> Result<Vec<NextTokenPrediction>> {
                Ok(vec![NextTokenPrediction{token:1, confidence:1.0}; 10]) // Too many
            }
        }
        struct EmptyDraft;
        impl DraftModel for EmptyDraft {
            fn draft(&mut self, _: &[TokenId], _: usize, _: Option<&mut (dyn crate::constraints::CfgMatcher + '_)>) -> Result<Vec<NextTokenPrediction>> {
                Ok(vec![])
            }
        }
        struct OkTarget;
        impl TargetModel for OkTarget {
            fn verify(&mut self, _: &[TokenId], d: &[TokenId], _: Option<&mut (dyn crate::constraints::CfgMatcher + '_)>) -> Result<Vec<VerifyStep>> {
                Ok(vec![VerifyStep{expected:1}; d.len()])
            }
        }
        struct BadTarget;
        impl TargetModel for BadTarget {
            fn verify(&mut self, _: &[TokenId], _: &[TokenId], _: Option<&mut (dyn crate::constraints::CfgMatcher + '_)>) -> Result<Vec<VerifyStep>> {
                Ok(vec![]) // Wrong length
            }
        }

        assert!(session.draft(&mut BadDraft, &mut OkTarget, 4).is_err());
        assert!(session.draft(&mut EmptyDraft, &mut OkTarget, 4).unwrap().is_empty());
       
        let mut good_draft = ScriptedDraft { script: vec![1, 2, 3] };
        assert!(session.draft(&mut good_draft, &mut BadTarget, 4).is_err());
    }

    #[test]
    fn test_dynamic_draft_window() {
        let decoder = SpeculativeDecoder::new(2).unwrap();
        let mut session = decoder.begin(&[1]).unwrap();
        assert_eq!(session.current_window(), 2);

        // 1. Full acceptance should grow window
        let drafted = vec![NextTokenPrediction { token: 2, confidence: 1.0 }, NextTokenPrediction { token: 3, confidence: 1.0 }];
        let verified = vec![VerifyStep { expected: 2 }, VerifyStep { expected: 3 }];
        session.commit(&drafted, &verified).unwrap();
        assert_eq!(session.current_window(), 3);

        // 2. Rejection should shrink window
        let drafted = vec![NextTokenPrediction { token: 4, confidence: 1.0 }, NextTokenPrediction { token: 9, confidence: 1.0 }];
        let verified = vec![VerifyStep { expected: 4 }, VerifyStep { expected: 5 }]; // 5 != 9
        session.commit(&drafted, &verified).unwrap();
        assert_eq!(session.current_window(), 2); // accepted=1 + 1 = 2
    }

    #[test]
    fn test_tree_best_path() {
        // Create a tree:
        //   root (0: token 10)
        //    /   \
        // child1(1: token 20)  child2(2: token 30)
        //  |
        // grandchild1(3: token 40)
        
        let mut tree = SpeculativeTree::new(10);
        let c1 = tree.add_child(0, 20);
        let _c2 = tree.add_child(0, 30);
        let _gc1 = tree.add_child(c1, 40);

        // Scenario 1: c2 is the only correct path
        let verified = TreeVerifyResult {
            expected: vec![30, 99, 99, 99], // After root (0), the target model says 30.
        };
        let (best_path, rejected) = tree.find_best_path(&verified);
        assert_eq!(best_path, vec![10, 30]);
        assert_eq!(rejected, Some(99)); // After 30 (node 2), target model says 99.
        
        // Scenario 2: c1 path is better
        let verified = TreeVerifyResult {
            expected: vec![20, 40, 99, 99], // root -> 20, 20 -> 40, 40 -> 99
        };
        let (best_path, rejected) = tree.find_best_path(&verified);
        assert_eq!(best_path, vec![10, 20, 40]);
        assert_eq!(rejected, Some(99));
    }
}
