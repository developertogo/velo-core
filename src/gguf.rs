//! Minimal GGUF file reader.
//!
//! Supports GGUF v1, v2, and v3. Parses metadata key-value pairs and the
//! tensor info table. Tensor data is returned as a raw `Vec<u8>` so the
//! caller controls memory.

use std::collections::HashMap;
use std::fmt;
use std::io::{self, Read, Seek, SeekFrom};

/// Magic bytes: ASCII "GGUF" as a little-endian u32.
const GGUF_MAGIC: u32 = 0x4647_4755;

/// Default tensor data alignment in bytes.
const DEFAULT_ALIGNMENT: u64 = 32;

// ── GGML quantization types ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    I8,
    I16,
    I32,
    I64,
    F64,
    BF16,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            30 => Self::BF16,
            _ => return None,
        })
    }

    /// (bytes_per_block, elements_per_block)
    pub fn block_params(self) -> (u64, u64) {
        match self {
            Self::F32 | Self::I32 => (4, 1),
            Self::F16 | Self::I16 | Self::BF16 => (2, 1),
            Self::Q4_0 => (18, 32),
            Self::Q4_1 => (20, 32),
            Self::Q5_0 => (22, 32),
            Self::Q5_1 => (24, 32),
            Self::Q8_0 => (34, 32),
            Self::Q8_1 => (36, 32),
            Self::Q2K => (84, 256),
            Self::Q3K => (110, 256),
            Self::Q4K => (144, 256),
            Self::Q5K => (176, 256),
            Self::Q6K => (210, 256),
            Self::Q8K => (292, 256),
            Self::I8 => (1, 1),
            Self::I64 | Self::F64 => (8, 1),
        }
    }

    /// Byte size for `n_elements` values of this type.
    pub fn nbytes(self, n_elements: u64) -> u64 {
        let (block_bytes, block_elems) = self.block_params();
        let n_blocks = n_elements.div_ceil(block_elems);
        n_blocks * block_bytes
    }
}

// ── Metadata value ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U64(v) => Some(*v),
            Self::U32(v) => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_i32(&self) -> Option<i32> {
        match self {
            Self::I32(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }
}

// ── Tensor info ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// Shape in row-major order (innermost dimension last).
    pub shape: Vec<u64>,
    pub dtype: GgmlType,
    /// Byte offset from the start of the data section.
    pub offset: u64,
    /// Total bytes this tensor occupies.
    pub nbytes: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.shape.iter().product()
    }
}

// ── GgufFile ──────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct GgufFile {
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, TensorInfo>,
    /// Absolute byte position in the source file where tensor data starts.
    pub data_offset: u64,
}

impl GgufFile {
    pub fn read<R: Read + Seek>(r: &mut R) -> Result<Self, GgufError> {
        // Magic
        let magic = read_u32(r)?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::BadMagic(magic));
        }

        let version = read_u32(r)?;
        if version == 0 || version > 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count = read_u64(r)?;
        let kv_count = read_u64(r)?;

        // Metadata key-value pairs
        let mut metadata = HashMap::with_capacity(kv_count as usize);
        for _ in 0..kv_count {
            let key = read_string(r)?;
            let value = read_value(r, version)?;
            metadata.insert(key, value);
        }

        // Tensor info
        let mut tensors = HashMap::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = read_string(r)?;
            let n_dims = read_u32(r)?;
            if n_dims > 4 {
                return Err(GgufError::Malformed(format!(
                    "tensor {name}: too many dims ({n_dims})"
                )));
            }
            let mut shape = vec![0u64; n_dims as usize];
            for d in &mut shape {
                *d = read_u64(r)?;
            }
            let type_id = read_u32(r)?;
            let dtype = GgmlType::from_u32(type_id)
                .ok_or(GgufError::UnknownType(type_id))?;
            let offset = read_u64(r)?;
            let n_elements: u64 = shape.iter().product();
            let nbytes = dtype.nbytes(n_elements);
            tensors.insert(
                name.clone(),
                TensorInfo { name, shape, dtype, offset, nbytes },
            );
        }

        // Data section starts after alignment padding
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_ALIGNMENT);

        let pos = r.stream_position()?;
        let data_offset = pos.div_ceil(alignment) * alignment;

        Ok(Self { version, metadata, tensors, data_offset })
    }

    /// Read the raw bytes of a named tensor from a seekable reader.
    pub fn read_tensor<R: Read + Seek>(
        &self,
        r: &mut R,
        name: &str,
    ) -> Result<Vec<u8>, GgufError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| GgufError::Malformed(format!("tensor not found: {name}")))?;
        r.seek(SeekFrom::Start(self.data_offset + info.offset))?;
        let mut buf = vec![0u8; info.nbytes as usize];
        r.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Read ALL tensor data into a single contiguous buffer.
    pub fn read_all_tensors<R: Read + Seek>(
        &self,
        r: &mut R,
    ) -> Result<Vec<u8>, GgufError> {
        let end = self
            .tensors
            .values()
            .map(|t| t.offset + t.nbytes)
            .max()
            .unwrap_or(0);
        r.seek(SeekFrom::Start(self.data_offset))?;
        let mut buf = vec![0u8; end as usize];
        r.read_exact(&mut buf)?;
        Ok(buf)
    }
}

// ── Error ─────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum GgufError {
    Io(io::Error),
    BadMagic(u32),
    UnsupportedVersion(u32),
    UnknownType(u32),
    Malformed(String),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::BadMagic(m) => write!(f, "bad magic: {m:#010x}"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported GGUF version {v}"),
            Self::UnknownType(t) => write!(f, "unknown ggml type {t}"),
            Self::Malformed(s) => write!(f, "malformed GGUF: {s}"),
        }
    }
}

impl std::error::Error for GgufError {}

impl From<io::Error> for GgufError {
    fn from(e: io::Error) -> Self { Self::Io(e) }
}

// ── Low-level readers ─────────────────────────────────────────────────────────

fn read_u8<R: Read>(r: &mut R) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}
fn read_i8<R: Read>(r: &mut R) -> io::Result<i8> { read_u8(r).map(|v| v as i8) }
fn read_u16<R: Read>(r: &mut R) -> io::Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}
fn read_i16<R: Read>(r: &mut R) -> io::Result<i16> { read_u16(r).map(|v| v as i16) }
fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}
fn read_i32<R: Read>(r: &mut R) -> io::Result<i32> { read_u32(r).map(|v| v as i32) }
fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}
fn read_i64<R: Read>(r: &mut R) -> io::Result<i64> { read_u64(r).map(|v| v as i64) }
fn read_f32<R: Read>(r: &mut R) -> io::Result<f32> {
    read_u32(r).map(f32::from_bits)
}
fn read_f64<R: Read>(r: &mut R) -> io::Result<f64> {
    read_u64(r).map(f64::from_bits)
}

fn read_string<R: Read>(r: &mut R) -> Result<String, GgufError> {
    let len = read_u64(r)?;
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf)
        .map_err(|e| GgufError::Malformed(format!("invalid UTF-8 string: {e}")))
}

fn read_value<R: Read>(r: &mut R, version: u32) -> Result<GgufValue, GgufError> {
    let type_id = read_u32(r)?;
    Ok(match type_id {
        0 => GgufValue::U8(read_u8(r)?),
        1 => GgufValue::I8(read_i8(r)?),
        2 => GgufValue::U16(read_u16(r)?),
        3 => GgufValue::I16(read_i16(r)?),
        4 => GgufValue::U32(read_u32(r)?),
        5 => GgufValue::I32(read_i32(r)?),
        6 => GgufValue::F32(read_f32(r)?),
        7 => GgufValue::Bool(read_u8(r)? != 0),
        8 => GgufValue::String(read_string(r)?),
        9 => {
            let elem_type = read_u32(r)?;
            let count = read_u64(r)?;
            let mut arr = Vec::with_capacity(count as usize);
            for _ in 0..count {
                // Arrays store raw values without a type tag (the tag was read above)
                arr.push(read_array_elem(r, elem_type, version)?);
            }
            GgufValue::Array(arr)
        }
        10 => GgufValue::U64(read_u64(r)?),
        11 => GgufValue::I64(read_i64(r)?),
        12 => GgufValue::F64(read_f64(r)?),
        t => return Err(GgufError::UnknownType(t)),
    })
}

/// Read a single array element — same as `read_value` minus the outer type tag.
fn read_array_elem<R: Read>(
    r: &mut R,
    type_id: u32,
    _version: u32,
) -> Result<GgufValue, GgufError> {
    Ok(match type_id {
        0 => GgufValue::U8(read_u8(r)?),
        1 => GgufValue::I8(read_i8(r)?),
        2 => GgufValue::U16(read_u16(r)?),
        3 => GgufValue::I16(read_i16(r)?),
        4 => GgufValue::U32(read_u32(r)?),
        5 => GgufValue::I32(read_i32(r)?),
        6 => GgufValue::F32(read_f32(r)?),
        7 => GgufValue::Bool(read_u8(r)? != 0),
        8 => GgufValue::String(read_string(r)?),
        10 => GgufValue::U64(read_u64(r)?),
        11 => GgufValue::I64(read_i64(r)?),
        12 => GgufValue::F64(read_f64(r)?),
        t => return Err(GgufError::UnknownType(t)),
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn write_u32(buf: &mut Vec<u8>, v: u32) { buf.extend_from_slice(&v.to_le_bytes()); }
    fn write_u64(buf: &mut Vec<u8>, v: u64) { buf.extend_from_slice(&v.to_le_bytes()); }
    fn write_str(buf: &mut Vec<u8>, s: &str) {
        write_u64(buf, s.len() as u64);
        buf.extend_from_slice(s.as_bytes());
    }

    fn minimal_gguf(version: u32) -> Vec<u8> {
        let mut b = Vec::new();
        write_u32(&mut b, GGUF_MAGIC);
        write_u32(&mut b, version);
        write_u64(&mut b, 0); // tensor_count
        write_u64(&mut b, 0); // kv_count
        b
    }

    #[test]
    fn parses_empty_v3_file() {
        let data = minimal_gguf(3);
        let file = GgufFile::read(&mut Cursor::new(&data)).unwrap();
        assert_eq!(file.version, 3);
        assert!(file.metadata.is_empty());
        assert!(file.tensors.is_empty());
    }

    #[test]
    fn rejects_bad_magic() {
        let mut data = minimal_gguf(3);
        data[0] = 0xFF;
        assert!(matches!(
            GgufFile::read(&mut Cursor::new(&data)),
            Err(GgufError::BadMagic(_))
        ));
    }

    #[test]
    fn rejects_unsupported_version() {
        let data = minimal_gguf(99);
        assert!(matches!(
            GgufFile::read(&mut Cursor::new(&data)),
            Err(GgufError::UnsupportedVersion(99))
        ));
    }

    #[test]
    fn parses_string_metadata() {
        let mut b = Vec::new();
        write_u32(&mut b, GGUF_MAGIC);
        write_u32(&mut b, 3);
        write_u64(&mut b, 0);   // tensors
        write_u64(&mut b, 1);   // 1 kv
        write_str(&mut b, "general.architecture");
        write_u32(&mut b, 8);   // STRING type
        write_str(&mut b, "llama");

        let file = GgufFile::read(&mut Cursor::new(&b)).unwrap();
        assert_eq!(
            file.metadata["general.architecture"].as_str(),
            Some("llama")
        );
    }

    #[test]
    fn parses_u32_metadata() {
        let mut b = Vec::new();
        write_u32(&mut b, GGUF_MAGIC);
        write_u32(&mut b, 3);
        write_u64(&mut b, 0);
        write_u64(&mut b, 1);
        write_str(&mut b, "llama.embedding_length");
        write_u32(&mut b, 4);   // U32
        write_u32(&mut b, 4096);

        let file = GgufFile::read(&mut Cursor::new(&b)).unwrap();
        assert_eq!(file.metadata["llama.embedding_length"].as_u32(), Some(4096));
    }

    #[test]
    fn parses_array_metadata() {
        let mut b = Vec::new();
        write_u32(&mut b, GGUF_MAGIC);
        write_u32(&mut b, 3);
        write_u64(&mut b, 0);
        write_u64(&mut b, 1);
        write_str(&mut b, "tokenizer.ggml.tokens");
        write_u32(&mut b, 9);   // ARRAY
        write_u32(&mut b, 8);   // elem type STRING
        write_u64(&mut b, 2);
        write_str(&mut b, "hello");
        write_str(&mut b, "world");

        let file = GgufFile::read(&mut Cursor::new(&b)).unwrap();
        let arr = file.metadata["tokenizer.ggml.tokens"].as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0].as_str(), Some("hello"));
        assert_eq!(arr[1].as_str(), Some("world"));
    }

    #[test]
    fn ggml_type_nbytes_q4_0() {
        // 32 elements → 1 block → 18 bytes
        assert_eq!(GgmlType::Q4_0.nbytes(32), 18);
        // 64 elements → 2 blocks → 36 bytes
        assert_eq!(GgmlType::Q4_0.nbytes(64), 36);
    }

    #[test]
    fn ggml_type_nbytes_f32() {
        assert_eq!(GgmlType::F32.nbytes(100), 400);
    }
}
