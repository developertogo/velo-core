use std::sync::atomic::{AtomicU32, Ordering};

/// Real-time hardware power draw telemetry.
pub trait PowerTelemetry: Send + Sync {
    /// Returns the current package power draw in milliwatts.
    fn current_power_mw(&self) -> u32;
    
    /// Returns the power budget cap in milliwatts.
    fn power_budget_mw(&self) -> u32;
}

/// A simulated SMC-based telemetry module for Apple Silicon.
pub struct SmcTelemetry {
    budget_mw: u32,
    simulated_draw: AtomicU32,
}

impl SmcTelemetry {
    pub fn new(budget_mw: u32) -> Self {
        Self {
            budget_mw,
            simulated_draw: AtomicU32::new(0),
        }
    }

    /// Update simulated power draw (for testing/mocking).
    pub fn set_simulated_draw(&self, mw: u32) {
        self.simulated_draw.store(mw, Ordering::Relaxed);
    }
}

impl PowerTelemetry for SmcTelemetry {
    fn current_power_mw(&self) -> u32 {
        #[cfg(target_os = "macos")]
        {
            // In a real implementation, we would call IOKit SMC keys like 'PSTR' or 'PCPT'
            // For now, we return the simulated draw.
            self.simulated_draw.load(Ordering::Relaxed)
        }
        #[cfg(not(target_os = "macos"))]
        {
            self.simulated_draw.load(Ordering::Relaxed)
        }
    }

    fn power_budget_mw(&self) -> u32 {
        self.budget_mw
    }
}

/// Dynamic KV-Cache and compute precision based on power headroom.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerState {
    /// Plenty of power budget. Max precision (FP16/FP32).
    Performance,
    /// Approaching budget limit. Balanced precision.
    Balanced,
    /// Hitting power budget. Lowest precision (INT8).
    Efficiency,
}

pub struct PrecisionGovernor {
    telemetry: Box<dyn PowerTelemetry>,
}

impl PrecisionGovernor {
    pub fn new(telemetry: Box<dyn PowerTelemetry>) -> Self {
        Self { telemetry }
    }

    /// Determines the optimal power state based on current draw vs budget.
    pub fn evaluate_state(&self) -> PowerState {
        let current = self.telemetry.current_power_mw();
        let budget = self.telemetry.power_budget_mw();

        let utilization = current as f32 / budget as f32;

        if utilization >= 0.90 {
            PowerState::Efficiency
        } else if utilization >= 0.70 {
            PowerState::Balanced
        } else {
            PowerState::Performance
        }
    }

    /// Adapts the KV cache type based on the current power state.
    pub fn adapt_kv_precision(&self, _current_type: crate::paged_attention::KvCacheType) -> crate::paged_attention::KvCacheType {
        match self.evaluate_state() {
            PowerState::Performance => crate::paged_attention::KvCacheType::Fp32,
            PowerState::Balanced => crate::paged_attention::KvCacheType::Fp8,
            PowerState::Efficiency => crate::paged_attention::KvCacheType::Int8,
        }
    }

    /// Adapts AMX precision based on power state.
    pub fn adapt_amx_precision(&self) -> crate::amx::AmxPrecision {
        match self.evaluate_state() {
            PowerState::Performance => crate::amx::AmxPrecision::Fp32,
            PowerState::Balanced => crate::amx::AmxPrecision::Bf16,
            PowerState::Efficiency => crate::amx::AmxPrecision::Fp16,
        }
    }

    /// Calculates a power-constrained batch size multiplier (0.0 to 1.0).
    pub fn batch_size_multiplier(&self) -> f32 {
        let current = self.telemetry.current_power_mw();
        let budget = self.telemetry.power_budget_mw();
        
        let utilization = current as f32 / budget as f32;
        if utilization > 1.0 {
            0.1 // severely throttle
        } else if utilization > 0.8 {
            0.5 // halve the batch size
        } else {
            1.0 // full batch size
        }
    }
}
