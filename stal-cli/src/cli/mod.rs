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
        save_path: PathBuf,
    },
    #[command(arg_required_else_help = true)]
    Classify {
        #[arg(help = "path to the model to be used")]
        /// The path to save the model.
        model: PathBuf,

        #[arg(help = "text to be classified")]
        /// The text to be classified.
        ///
        /// Accept only pure texts in several ways:
        ///
        /// 1. **Text File**. The most recommended way is to pass a pure text file (`.txt`). It will be read and classified.
        /// 2. **String**. You can also pass a string directly as the text. This is useful when the text is very short.
        /// 3. **Text Stream**. You can pass a text stream as the text. This is useful when the text is the output of another command.
        /// 4. **Direct Input**. If nothing passed, you'll be required to type in the text in the commandline.
        ///
        /// ## Examples
        ///
        /// 1. `stal-cli classify model.postcard text.txt`
        /// 2. `stal-cli classify model.postcard "This is the text to classify"`
        /// 3. `echo text.txt | stal-cli classify model.postcard`
        /// 4. `stal-cli classify model.postcard` and the commandline should require the use to type some texts
        text: Option<String>,

        #[arg(long, help = "Uses rich output", conflicts_with = "concise")]
        /// Uses rich output.
        ///
        /// This outputs the classified result in a more detailed way, showing the classification of
        /// each sentence and some analysis to those result.
        rich: bool,

        #[arg(long, help = "Uses concise output", conflicts_with = "rich")]
        /// Uses concise output.
        ///
        /// This outputs the classified result in a more concise way, showing just the most familiar
        /// author and the possibility.
        concise: bool,
    },
}

impl Cli {
    pub fn execute() -> anyhow::Result<()> {
        let cli = Cli::parse();
        match cli.command {
            Commands::Train { dataset, save_path } => {
                log::trace!("`train` command.");
                train(dataset, save_path)
            }
            Commands::Classify {
                model,
                text,
                rich,
                concise,
            } => {
                log::trace!("`classify` command.");
                classify(model, text, rich, concise)
            }
        }
    }
}
