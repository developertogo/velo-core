use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::path::{Path, PathBuf};
use tokio::sync::oneshot;

use crate::metal::model::LlamaMetalModel;
use crate::metal::runtime::MetalRuntimeHandles;
use crate::model_loader::load_gguf;
use crate::speculative::{Result, SpeculativeError};

/// A pool of models that are resident in memory (CPU or GPU).
///
/// This structure allows for zero-latency switching between models by keeping
/// their weights and pipelines pre-loaded.
pub struct ModelPool {
    models: Arc<Mutex<HashMap<String, Arc<Mutex<LlamaMetalModel>>>>>,
    handles: MetalRuntimeHandles,
}

impl ModelPool {
    /// Creates a new, empty model pool.
    pub fn new(handles: MetalRuntimeHandles) -> Self {
        Self {
            models: Arc::new(Mutex::new(HashMap::new())),
            handles,
        }
    }

    /// Returns a model by name if it exists in the pool.
    pub fn get(&self, name: &str) -> Option<Arc<Mutex<LlamaMetalModel>>> {
        self.models.lock().unwrap().get(name).cloned()
    }

    /// Adds a model to the pool.
    pub fn add(&self, name: String, model: LlamaMetalModel) {
        self.models.lock().unwrap().insert(name, Arc::new(Mutex::new(model)));
    }

    /// Removes a model from the pool.
    pub fn remove(&self, name: &str) {
        self.models.lock().unwrap().remove(name);
    }

    /// Prefetches a model from a GGUF file in the background.
    ///
    /// This method returns immediately and provides a receiver that will
    /// signal when the model is ready.
    pub fn prefetch(&self, name: String, path: PathBuf) -> oneshot::Receiver<Result<()>> {
        let (tx, rx) = oneshot::channel();
        let models = self.models.clone();
        let handles = self.handles.clone();
        let name_clone = name.clone();

        std::thread::spawn(move || {
            let res = (|| -> Result<()> {
                let device = handles.device.as_ref().ok_or_else(|| SpeculativeError::Model("No device".into()))?;
                let queue = handles.command_queue.as_ref().ok_or_else(|| SpeculativeError::Model("No queue".into()))?;
                let library = handles.library.as_ref().ok_or_else(|| SpeculativeError::Model("No library".into()))?;

                // 1. Load GGUF from disk
                let store = load_gguf(&path).map_err(|e| SpeculativeError::Model(e.to_string()))?;
                
                // 2. Initialize Metal model and upload weights
                let mut model = LlamaMetalModel::new(store.meta.clone(), device.clone(), queue.clone(), library.clone());
                model.upload_weights(&store)?;

                // 3. Add to pool
                models.lock().unwrap().insert(name_clone, Arc::new(Mutex::new(model)));
                Ok(())
            })();

            let _ = tx.send(res);
        });

        rx
    }

    /// Loads a model synchronously (blocking).
    pub fn load_sync(&self, name: String, path: &Path) -> Result<()> {
        let device = self.handles.device.as_ref().ok_or_else(|| SpeculativeError::Model("No device".into()))?;
        let queue = self.handles.command_queue.as_ref().ok_or_else(|| SpeculativeError::Model("No queue".into()))?;
        let library = self.handles.library.as_ref().ok_or_else(|| SpeculativeError::Model("No library".into()))?;

        let store = load_gguf(path).map_err(|e| SpeculativeError::Model(e.to_string()))?;
        let mut model = LlamaMetalModel::new(store.meta.clone(), device.clone(), queue.clone(), library.clone());
        model.upload_weights(&store)?;

        self.models.lock().unwrap().insert(name, Arc::new(Mutex::new(model)));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_pool_basic() {
        let handles = MetalRuntimeHandles {
            device: None,
            command_queue: None,
            library: None,
        };
        let pool = ModelPool::new(handles);
        assert!(pool.get("test").is_none());
        
        pool.remove("test"); // Should not panic
    }

    #[test]
    fn test_model_pool_errors() {
        let handles = MetalRuntimeHandles {
            device: None,
            command_queue: None,
            library: None,
        };
        let pool = ModelPool::new(handles);
        let rx = pool.prefetch("test".into(), PathBuf::from("fake.gguf"));
        
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let res = rx.await.unwrap();
            assert!(res.is_err());
            assert!(format!("{:?}", res.err().unwrap()).contains("No device"));
        });
        
        let res_sync = pool.load_sync("test".into(), Path::new("fake.gguf"));
        assert!(res_sync.is_err());
        assert!(format!("{:?}", res_sync.err().unwrap()).contains("No device"));
    }
}
