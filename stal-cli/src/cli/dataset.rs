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
    let json = std::fs::read_to_string(dataset)?;
    let json_dataset: Vec<JsonDataset> = serde_json::from_str(&json)?;
    let dataset = json_dataset
        .into_iter()
        .map(|json_dataset| (json_dataset.author, json_dataset.text_path))
        .collect();
    Ok(dataset)
}

pub fn load_dir_dataset<P: AsRef<Path>>(dir: P) -> Result<Vec<(String, String)>, DatasetLoadError> {
    let path = dir.as_ref().normalize()?;

    let mut dataset = vec![];
    let mut authors = WalkDir::new(path).into_iter();
    authors.next();
    for author_entry in authors {
        let entry = author_entry?;
        log::debug!("Load Entry(Author): {:?}", entry.path());
        let author = entry
            .file_name()
            .to_str()
            .ok_or_else(|| DatasetLoadError::InvalidAuthorName(entry.path().to_path_buf()))?
            .to_string();
        let mut texts = WalkDir::new(entry.path()).into_iter();
        texts.next();
        for text in texts {
            let text = text?;
            log::debug!("Load Text: {:?}", text.path());
            let path = text.path().to_path_buf();
            if let Some(ext) = path.extension() {
                if ext == "txt" {
                    if let Some(path) = path.to_str() {
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
