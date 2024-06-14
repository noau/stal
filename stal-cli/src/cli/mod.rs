use std::path::PathBuf;

use clap::{Parser, Subcommand};

use crate::cli::model::{classify, train};

mod dataset;
mod model;

#[derive(Debug, Parser)]
#[command(name = "stal-cli")]
#[command(about = "A cli tool for stylish analysis", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    #[command(arg_required_else_help = true)]
    Train {
        #[arg(
            value_name = "DIRECTORY|JSON",
            help = "dataset used to train the model"
        )]
        /// Dataset used to train the model
        ///
        /// There are two ways to specify dataset.
        ///
        /// 1. **Directory**. Specify a directory that contains subdirectories. Each subdirectory is a class, i.e., an author. Under that subdirectory, texts to be used must be pure texts, ends with `.txt`.
        /// 2. **JSON**. A `JSON` array could be used to specify the data. Each element must be a `JSON` object with two properties: `text_path` and `author`.
        dataset: String,
        #[arg(help = "path to save the model")]
        /// The path to save the model.
        ///
        /// Recommended suffix is `.postcard`, and `.model` is also acceptable.
        ///
        save_path: PathBuf,
    },
    #[command(arg_required_else_help = true)]
    Classify {
        #[arg(help = "path to the model to be used")]
        model: PathBuf,

        #[arg(help = "text to be classified")]
        text: Option<String>,

        #[arg(long, help = "Uses rich output", conflicts_with = "concise")]
        rich: bool,

        #[arg(long, help = "Uses concise output", conflicts_with = "rich")]
        concise: bool,
    },
}

impl Cli {
    pub fn execute() -> anyhow::Result<()> {
        let cli = Cli::parse();
        match cli.command {
            Commands::Train { dataset, save_path } => train(dataset, save_path),
            Commands::Classify {
                model,
                text,
                rich,
                concise,
            } => classify(model, text, rich, concise),
        }
    }
}
