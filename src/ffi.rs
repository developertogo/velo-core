use std::ffi::CStr;
use std::os::raw::c_char;
use std::ptr;
use crate::engine::{VeloEngine, EngineConfig};
use crate::metal::{MetalMemoryRuntime, MetalRuntimeConfig, Quantization};
use crate::runtime::MemoryRuntimeConfig;
use crate::paged_attention::KvCacheType;

#[repr(C)]
pub struct VeloEngineHandle {
    _private: [u8; 0],
}

#[unsafe(no_mangle)]
pub extern "C" fn velo_core_engine_new(
    model_name_ptr: *const c_char,
    max_slots: usize,
    max_context_tokens: usize,
) -> *mut VeloEngineHandle {
    if model_name_ptr.is_null() {
        return ptr::null_mut();
    }

    let model_name = unsafe {
        match CStr::from_ptr(model_name_ptr).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return ptr::null_mut(),
        }
    };

    let config = EngineConfig {
        draft_window: 8,
        memory: MemoryRuntimeConfig::cpu(max_context_tokens, 16, 32, max_slots, max_slots),
        kv_type: KvCacheType::Fp32, // Default for Metal usually
    };

    let runtime_config = MetalRuntimeConfig {
        model_name,
        memory: config.memory,
        quantization: Quantization::Q4_0, // Default to Q4_0 for efficiency
    };

    match MetalMemoryRuntime::new(runtime_config) {
        Ok(runtime) => {
            match VeloEngine::with_runtime(config, runtime) {
                Ok(engine) => Box::into_raw(Box::new(engine)) as *mut VeloEngineHandle,
                Err(_) => ptr::null_mut(),
            }
        }
        Err(_) => ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn velo_core_engine_free(engine_ptr: *mut VeloEngineHandle) {
    if !engine_ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(engine_ptr as *mut VeloEngine<MetalMemoryRuntime>);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn velo_core_engine_generate(
    engine_ptr: *mut VeloEngineHandle,
    prompt_ptr: *const u32,
    prompt_len: usize,
    _max_new_tokens: usize,
    output_len_ptr: *mut usize,
) -> *mut u32 {
    if engine_ptr.is_null() || prompt_ptr.is_null() || output_len_ptr.is_null() {
        return ptr::null_mut();
    }

    let _engine = unsafe { &mut *(engine_ptr as *mut VeloEngine<MetalMemoryRuntime>) };
    let prompt = unsafe { std::slice::from_raw_parts(prompt_ptr, prompt_len) };

    // POC: Just return the prompt back as "generated" tokens to verify the bridge works.
    let mut result = prompt.to_vec();
    unsafe { *output_len_ptr = result.len() };
    
    let ptr = result.as_mut_ptr();
    std::mem::forget(result);
    ptr
}

#[unsafe(no_mangle)]
pub extern "C" fn velo_core_free_tokens(tokens_ptr: *mut u32, len: usize) {
    if !tokens_ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(tokens_ptr, len, len);
        }
    }
}
