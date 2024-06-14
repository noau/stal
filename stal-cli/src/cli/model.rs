use std::io::{IsTerminal, Read};
use std::path::PathBuf;

use crate::cli::dataset::{load_dir_dataset, load_json_dataset};

/// Train the model
pub fn train(dataset: String, save_path: PathBuf) -> anyhow::Result<()> {
    log::trace!("Current dir: {:?}", std::env::current_dir()?);
    let dataset = if dataset.ends_with(".json") {
        log::trace!("Load dataset according to JSON configuration.");
        load_json_dataset(dataset)
    } else {
        log::trace!("Load dataset from nested directory.");
        load_dir_dataset(dataset)
    }?;
    log::info!("Start training the model.");
    let model = stal_core::model::BayesianModel::train(dataset)?;
    log::info!("Model training finished: {}.", model);

    log::info!("Saving model to `{}`", save_path.display());
    // Ensure that the path used for saving model exists.
    fsio::file::ensure_exists(&save_path)?;
    model.save(&save_path)?;
    log::info!("Model saved.");
    Ok(())
}

/// Classify using the specified model
pub fn classify(
    model: PathBuf,
    text: Option<String>,
    _rich: bool,
    _concise: bool,
) -> anyhow::Result<()> {
    log::trace!("Get the input text.");
    let input = get_input(text)?;
    log::trace!("Load specified model.");
    let model = stal_core::model::BayesianModel::load(model)?;
    log::trace!("Start classification.");
    let result = model.classify_text(&input);
    // TODO: `rich` and `concise` output
    // TODO: Format classification result
    println!("{:#?}", result);
    Ok(())
}

pub fn get_input(option: Option<String>) -> anyhow::Result<String> {
    if let Some(str) = option {
        if str.ends_with(".txt") {
            log::trace!("Read texts from file: `{}`.", str);
            Ok(std::fs::read_to_string(str)?)
        } else {
            log::trace!("Uses cli argument as text: '{}'.", str);
            Ok(str)
        }
    } else {
        let mut buffer = String::new();
        if std::io::stdin().is_terminal() {
            log::trace!("Read texts from cmd.");
            println!("Please enter the text:");
            std::io::stdin().read_line(&mut buffer)?;
        } else {
            log::trace!("Read texts from pipeline.");
            std::io::stdin().read_to_string(&mut buffer)?;
        }
        Ok(buffer)
    }
}
