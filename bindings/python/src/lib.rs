use std::collections::HashMap;
use std::str;

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use rwkv_tokenizer;

#[derive(Debug)]
#[pyclass]
pub(crate) struct WorldTokenizer {
    tokenizer: rwkv_tokenizer::WorldTokenizer,
}

#[pymethods]
impl WorldTokenizer {
    #[new]
    pub(crate) fn new(filename: &str) -> PyResult<WorldTokenizer> {
        let tokenizer = rwkv_tokenizer::WorldTokenizer::new(Option::from(filename))
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        Ok(WorldTokenizer { tokenizer })
    }

    #[staticmethod]
    pub(crate) fn from_buffer(buffer: &[u8]) -> PyResult<WorldTokenizer> {
        let tokenizer = rwkv_tokenizer::WorldTokenizer::from_buffer(buffer)
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        Ok(WorldTokenizer { tokenizer })
    }

    pub(crate) fn encode(&self, word: &str) -> Vec<u16> {
        self.tokenizer.encode(word)
    }

    pub(crate) fn encode_batch(&self, word_list: Vec<String>) -> Vec<Vec<u16>> {
        self.tokenizer.encode_batch(word_list)
    }

    pub(crate) fn decode(&self, vec: Vec<u16>) -> PyResult<String> {
        self.tokenizer
            .decode(vec)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    pub(crate) fn vocab_size(&self) -> usize {
        return self.tokenizer.vocab_size();
    }

    pub(crate) fn get_vocab(&self) -> HashMap<String, usize> {
        return self.tokenizer.get_vocab();
    }
}

#[pymodule]
fn pyrwkv_tokenizer(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<WorldTokenizer>()?;
    Ok(())
}
