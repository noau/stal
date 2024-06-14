use std::io::{IsTerminal, Read};
use std::path::PathBuf;

use crate::cli::dataset::{load_dir_dataset, load_json_dataset};

pub fn train(dataset: String, save_path: PathBuf) -> anyhow::Result<()> {
    log::trace!("Current dir: {:?}", std::env::current_dir()?);
    let dataset = if dataset.ends_with(".json") {
        log::trace!("Load dataset according to JSON configuration.");
        load_json_dataset(dataset)
    } else {
        log::trace!("Load dataset from nested directory.");
        load_dir_dataset(dataset)
    }?;
    fsio::file::ensure_exists(&save_path)?;
    let model = stal_core::model::BayesianModel::train(dataset)?;
    model.save(&save_path)?;
    Ok(())
}

pub fn classify(
    model: PathBuf,
    text: Option<String>,
    _rich: bool,
    _concise: bool,
) -> anyhow::Result<()> {
    let input = get_input(text)?;
    let model = stal_core::model::BayesianModel::load(model)?;
    let result = model.classify_text(&input);
    println!("{:#?}", result);
    Ok(())
}

pub fn get_input(option: Option<String>) -> anyhow::Result<String> {
    if let Some(str) = option {
        if str.ends_with(".txt") {
            Ok(std::fs::read_to_string(str)?)
        } else {
            Ok(str)
        }
    } else {
        let mut buffer = String::new();
        if std::io::stdin().is_terminal() {
            println!("Please enter the text:");
            std::io::stdin().read_line(&mut buffer)?;
        } else {
            std::io::stdin().read_to_string(&mut buffer)?;
        }
        Ok(buffer)
    }
}
