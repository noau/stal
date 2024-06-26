use std::path::{Path, PathBuf};

use normpath::PathExt;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use walkdir::WalkDir;

#[derive(Debug, Error)]
pub enum DatasetLoadError {
    #[error("Failed to read file.")]
    Io(#[from] std::io::Error),
    #[error("Failed to parse JSON dataset.")]
    ParseJson(#[from] serde_json::Error),
    #[error("Failed to load the directory.")]
    Dir(#[from] walkdir::Error),
    #[error("Invalid directory name: {0}")]
    InvalidAuthorName(PathBuf),
    #[error("Invalid file name: {0}")]
    InvalidFileName(PathBuf),
    #[error("Unsupported file format: {0}")]
    UnsupportedFormat(PathBuf),
}

#[derive(Serialize, Deserialize)]
struct JsonDataset {
    pub text_path: String,
    pub author: String,
}

pub fn load_json_dataset<P>(dataset: P) -> Result<Vec<(String, String)>, DatasetLoadError>
where
    P: AsRef<Path>,
{
    log::trace!("Read dataset json from: `{}`.", dataset.as_ref().display());
    let json = std::fs::read_to_string(dataset)?;
    log::trace!("Parse dataset.");
    let json_dataset: Vec<JsonDataset> = serde_json::from_str(&json)?;
    let dataset = json_dataset
        .into_iter()
        .map(|json_dataset| (json_dataset.author, json_dataset.text_path))
        .collect();
    Ok(dataset)
}

pub fn load_dir_dataset<P: AsRef<Path>>(dir: P) -> Result<Vec<(String, String)>, DatasetLoadError> {
    log::trace!("Normalize path: `{}`.", dir.as_ref().display());
    let path = dir.as_ref().normalize()?;

    let mut dataset = vec![];
    log::trace!("Iter all authors.");
    let mut authors = WalkDir::new(path).into_iter();
    authors.next(); // Skip self
    for author_entry in authors {
        let entry = author_entry?;
        log::trace!("Load Entry(Author): {:?}", entry.path());
        let author = entry
            .file_name()
            .to_str()
            .ok_or_else(|| DatasetLoadError::InvalidAuthorName(entry.path().to_path_buf()))?
            .to_string();
        log::trace!("Iter all texts of '{}'.", author);
        let mut texts = WalkDir::new(entry.path()).into_iter();
        texts.next(); // Skip self
        for text in texts {
            let text = text?;
            log::trace!("Load Text: {:?}", text.path());
            let path = text.path().to_path_buf();
            if let Some(ext) = path.extension() {
                if ext == "txt" {
                    if let Some(path) = path.to_str() {
                        log::trace!("Find text: `{}`.", path);
                        dataset.push((author.clone(), path.to_string()));
                    } else {
                        return Err(DatasetLoadError::InvalidFileName(path));
                    }
                }
            } else {
                return Err(DatasetLoadError::UnsupportedFormat(path));
            }
        }
    }

    Ok(dataset)
}
