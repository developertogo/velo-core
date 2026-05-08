use crate::radix_cache::TokenId;
use crate::speculative::NextTokenPrediction;

/// A mask of allowed tokens.
pub type LogitMask = vob::Vob<usize>;

#[derive(Debug, Clone, Copy, PartialEq)]
struct TokenLogit {
    token: TokenId,
    logit: f32,
}

impl Eq for TokenLogit {}

impl PartialOrd for TokenLogit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.logit.partial_cmp(&other.logit)
    }
}

impl Ord for TokenLogit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

pub trait Sampler {
    fn sample(&self, logits: &[f32], mask: Option<&LogitMask>) -> NextTokenPrediction;
    fn is_greedy(&self) -> bool { false }
}

pub struct GreedySampler;

impl Sampler for GreedySampler {
    fn sample(&self, logits: &[f32], mask: Option<&LogitMask>) -> NextTokenPrediction {
        let mut max_logit = -f32::INFINITY;
        let mut max_token = 0;
        for (i, &logit) in logits.iter().enumerate() {
            if let Some(mask) = mask {
                if !mask.get(i).unwrap_or(false) {
                    continue;
                }
            }
            if logit >= max_logit {
                max_logit = logit;
                max_token = i as TokenId;
            }
        }
        NextTokenPrediction {
            token: max_token,
            confidence: max_logit,
        }
    }
    fn is_greedy(&self) -> bool { true }
}

pub struct TopPSampler {
    pub p: f32,
    pub temperature: f32,
}

impl Sampler for TopPSampler {
    fn sample(&self, logits: &[f32], mask: Option<&LogitMask>) -> NextTokenPrediction {
        let temp = self.temperature.max(1e-6);
        if self.p >= 1.0 && self.temperature <= 0.0 {
            return GreedySampler.sample(logits, mask);
        }

        let mut token_logits: Vec<TokenLogit> = logits
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                if let Some(mask) = mask {
                    mask.get(*i).unwrap_or(false)
                } else {
                    true
                }
            })
            .map(|(i, &l)| TokenLogit {
                token: i as TokenId,
                logit: l / temp,
            })
            .collect();
        
        if token_logits.is_empty() {
             return GreedySampler.sample(logits, None); // Fallback
        }

        // Sort descending
        token_logits.sort_by(|a, b| b.logit.partial_cmp(&a.logit).unwrap());

        // Softmax for top-p selection
        let max_logit = token_logits[0].logit;
        let exps: Vec<f32> = token_logits.iter().map(|tl| (tl.logit - max_logit).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        
        let mut cumulative_p = 0.0;
        let mut cutoff_idx = token_logits.len();
        for (i, &e) in exps.iter().enumerate() {
            let prob = e / sum_exp;
            cumulative_p += prob;
            if cumulative_p >= self.p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Renormalize and random sample
        let sub_sum: f32 = exps[..cutoff_idx].iter().sum();
        let r: f32 = rand::random::<f32>();
        let mut current_p = 0.0;
        for i in 0..cutoff_idx {
            let p_val = exps[i] / sub_sum;
            current_p += p_val;
            if r <= current_p {
                return NextTokenPrediction {
                    token: token_logits[i].token,
                    confidence: token_logits[i].logit,
                };
            }
        }

        NextTokenPrediction {
            token: token_logits[0].token,
            confidence: token_logits[0].logit,
        }
    }
}

pub struct MinPSampler {
    pub min_p: f32,
    pub temperature: f32,
}

impl Sampler for MinPSampler {
    fn sample(&self, logits: &[f32], mask: Option<&LogitMask>) -> NextTokenPrediction {
        let temp = self.temperature.max(1e-6);
        let mut token_logits: Vec<TokenLogit> = logits
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                if let Some(mask) = mask {
                    mask.get(*i).unwrap_or(false)
                } else {
                    true
                }
            })
            .map(|(i, &l)| TokenLogit {
                token: i as TokenId,
                logit: l / temp,
            })
            .collect();

        if token_logits.is_empty() {
            return GreedySampler.sample(logits, None); // Fallback
        }

        token_logits.sort_by(|a, b| b.logit.partial_cmp(&a.logit).unwrap());

        let max_logit = token_logits[0].logit;
        let cutoff = self.min_p; // min_p is relative to max prob

        let mut filtered_exps = Vec::new();
        let mut filtered_tokens = Vec::new();
        let mut sum_exp = 0.0;
        
        for tl in token_logits {
            let p_rel = (tl.logit - max_logit).exp();
            if p_rel < cutoff && !filtered_exps.is_empty() {
                break;
            }
            filtered_exps.push(p_rel);
            filtered_tokens.push((tl.token, tl.logit));
            sum_exp += p_rel;
        }

        let r: f32 = rand::random::<f32>();
        let mut current_p = 0.0;
        for (i, &e) in filtered_exps.iter().enumerate() {
            current_p += e / sum_exp;
            if r <= current_p {
                return NextTokenPrediction {
                    token: filtered_tokens[i].0,
                    confidence: filtered_tokens[i].1,
                };
            }
        }

        NextTokenPrediction {
            token: filtered_tokens[0].0,
            confidence: filtered_tokens[0].1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_sampler() {
        let logits = vec![1.0, 5.0, 2.0, 4.0];
        let sampler = GreedySampler;
        let res = sampler.sample(&logits, None);
        assert_eq!(res.token, 1);
        assert_eq!(res.confidence, 5.0);
        assert!(sampler.is_greedy());
    }

    #[test]
    fn test_greedy_sampler_with_mask() {
        let logits = vec![1.0, 5.0, 2.0, 4.0];
        let mut mask = vob::vob![false; 4];
        mask.set(0, true);
        mask.set(2, true);
        
        let sampler = GreedySampler;
        let res = sampler.sample(&logits, Some(&mask));
        assert_eq!(res.token, 2);
        assert_eq!(res.confidence, 2.0);
    }

    #[test]
    fn test_top_p_sampler_greedy_fallback() {
        let logits = vec![1.0, 5.0, 2.0, 4.0];
        let sampler = TopPSampler { p: 1.0, temperature: 0.0 };
        let res = sampler.sample(&logits, None);
        assert_eq!(res.token, 1);
    }

    #[test]
    fn test_top_p_sampler_basic() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let sampler = TopPSampler { p: 0.5, temperature: 1.0 };
        let res = sampler.sample(&logits, None);
        // With p=0.5, only tokens 3 and maybe 2 will be in the pool.
        assert!(res.token == 3 || res.token == 2);
    }

    #[test]
    fn test_min_p_sampler_basic() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let sampler = MinPSampler { min_p: 0.1, temperature: 1.0 };
        let res = sampler.sample(&logits, None);
        assert!(res.token <= 3);
    }
    
    #[test]
    fn test_samplers_empty_fallback() {
        let logits = vec![1.0, 2.0];
        let mask = vob::vob![false; 2];
        
        let top_p = TopPSampler { p: 0.9, temperature: 1.0 };
        assert_eq!(top_p.sample(&logits, Some(&mask)).token, 1); // Falls back to greedy max

        let min_p = MinPSampler { min_p: 0.1, temperature: 1.0 };
        assert_eq!(min_p.sample(&logits, Some(&mask)).token, 1);
    }

    #[test]
    fn test_token_logit_ord() {
        let a = TokenLogit { token: 1, logit: 1.0 };
        let b = TokenLogit { token: 2, logit: 2.0 };
        assert!(b > a);
        assert_eq!(a.cmp(&b), std::cmp::Ordering::Less);
    }
}
