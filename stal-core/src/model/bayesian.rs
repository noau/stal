use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter};
use std::io;
use std::path::Path;

use charabia::Segment;
use serde::{Deserialize, Serialize};
use text_splitter::TextSplitter;
use thiserror::Error;

#[derive(Debug)]
pub struct Predication {
    pub sentences_predicate: Vec<(usize, Vec<f32>)>,
    pub total_predicate: Vec<f32>,
}

#[derive(Debug)]
pub struct Classification {
    pub sentences_classification: Vec<(usize, HashMap<String, f32>)>,
    pub total_classification: HashMap<String, f32>,
}

#[derive(Debug, Error)]
pub enum BayesianSaveError {
    #[error("Failed to serialize model.")]
    Serialization(#[from] postcard::Error),
    #[error("Failed to create file")]
    File(#[from] fsio::error::FsIOError),
    #[error("Failed to write model into file.")]
    IO(#[from] io::Error),
}

#[derive(Debug, Error)]
pub enum BayesianLoadError {
    #[error("Failed to deserialize model.")]
    Deserialization(#[from] postcard::Error),
    #[error("Failed to read model from file.")]
    IO(#[from] io::Error),
}

#[derive(Debug, Serialize, Deserialize)]
/// A smoothed naive bayes model for Stylish Analysis
pub struct BayesianModel {
    /// List of authors
    authors: Vec<String>,
    /// Token count in each author's text(s) of each token
    token_author_dict: HashMap<String, Vec<u32>>,
    /// Each author's total token count
    author_token_count: Vec<u32>,
    /// Total token count used to train the model
    total_token_count: u32,
}

const MAX_SENTENCE_LENGTH: usize = 96;

/// Rating of un-seen token
const NO_TOKEN_RATING: f32 = 0.4;

const MIN_RATING: f32 = 0.2;

const MAX_RATING: f32 = 0.7;

impl BayesianModel {
    /// Train the bayesian model using given dataset. The dataset consists of `String` pairs, where
    /// the first is author, and the second is path to the text file. Must be `.txt` format of pure text.
    pub fn train(dataset: Vec<(String, String)>) -> io::Result<Self> {
        log::trace!("Find all authors.");
        let author_dict = dataset
            .iter()
            .map(|(author, _)| author.clone())
            .collect::<HashSet<_>>();
        let authors = author_dict.into_iter().collect::<Vec<_>>();
        let author_count = authors.len();
        log::trace!("Contains {} authors in total.", author_count);

        // Find all words and their count in each author's texts
        let mut token_author_dict: HashMap<String, Vec<u32>> = HashMap::new();
        for (author, path) in dataset {
            log::trace!("Indexing: ('{}', `{}`)", author, path);
            let author_index = authors.iter().position(|name| author.eq(name)).unwrap();

            let text = std::fs::read_to_string(path)?;
            let tokens = Self::tokenize(&text);
            for token in tokens {
                let token = token.to_string();
                let token_count = token_author_dict
                    .entry(token)
                    .or_insert(vec![0; author_count]);
                token_count[author_index] += 1;
            }
        }

        let author_token_count =
            token_author_dict
                .iter()
                .fold(vec![0; author_count], |acc, (_, token_count)| {
                    acc.iter()
                        .zip(token_count.iter())
                        .map(|(a, b)| *a + *b)
                        .collect::<Vec<_>>()
                });
        let total_token_count = author_token_count.iter().sum::<u32>();
        log::trace!("Contains {} tokens in total.", total_token_count);
        Ok(Self {
            authors,
            author_token_count,
            token_author_dict,
            total_token_count,
        })
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), BayesianSaveError> {
        fsio::file::ensure_exists(&path.as_ref())?;
        let bin_vec = postcard::to_allocvec(self)?;
        std::fs::write(path, bin_vec)?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, BayesianLoadError> {
        let bin_vec = std::fs::read(path)?;
        let model = postcard::from_bytes(&bin_vec)?;
        Ok(model)
    }

    pub fn preprocess(text: &str) -> Vec<(usize, Vec<&str>)> {
        let ts = TextSplitter::new(MAX_SENTENCE_LENGTH);
        ts.chunk_indices(text)
            .map(|(index, sentence)| (index, Self::tokenize(sentence)))
            .collect::<Vec<_>>()
    }

    pub fn tokenize(text: &str) -> Vec<&str> {
        // TODO: Remove punctuations and meaningless words
        text.segment_str().collect::<Vec<_>>()
    }

    pub fn classify_text(&self, text: &str) -> Classification {
        // Add the author name to the predication
        let transform = |v: Vec<f32>| {
            v.into_iter()
                .enumerate()
                .map(|(author, probability)| (self.authors[author].clone(), probability))
                .collect()
        };
        let Predication {
            sentences_predicate,
            total_predicate,
        } = self.predicate_text(text);
        let sentences_classification = sentences_predicate
            .into_iter()
            .map(|(sentence_index, predication)| (sentence_index, transform(predication)))
            .collect::<Vec<_>>();
        let total_classification = transform(total_predicate);
        Classification {
            sentences_classification,
            total_classification,
        }
    }

    fn predicate_text(&self, text: &str) -> Predication {
        let sentences = Self::preprocess(text);
        let sentence_count = sentences.len();
        let sentences_predicate = sentences
            .into_iter()
            .map(|(sentence_index, sentence)| (sentence_index, self.predicate(&sentence)))
            .collect::<Vec<_>>();
        let author_count = self.authors.len();
        let total_predicate = sentences_predicate
            .iter()
            .map(|(_, sentence_probability)| sentence_probability)
            .fold(vec![0.0; author_count], |acc, word_probability| {
                acc.iter()
                    .zip(word_probability.iter())
                    .map(|(&a, &b)| a + b)
                    .collect::<Vec<_>>()
            });
        // Normalize
        let total_predicate = total_predicate
            .into_iter()
            .map(|probability| probability / sentence_count as f32)
            .collect();
        Predication {
            sentences_predicate,
            total_predicate,
        }
    }

    fn predicate(&self, tokens: &Vec<&str>) -> Vec<f32> {
        let author_count = self.authors.len();
        let mut ratings = vec![vec![]; author_count];

        for token in tokens {
            let token = token.to_string();
            if let Some(token_count) = self.token_author_dict.get(&token) {
                let count = token_count.iter().sum::<u32>();
                for author in 0..author_count {
                    let token_author_count = token_count[author];
                    if token_author_count == 0 {
                        ratings[author].push(NO_TOKEN_RATING);
                    }

                    let this_probability =
                        token_author_count as f32 / self.author_token_count[author] as f32;
                    let other_probability = (count - token_author_count) as f32
                        / (self.total_token_count - self.author_token_count[author]) as f32;
                    // Clamp the rating to remove extreme ones
                    let rating = (this_probability / (this_probability + other_probability))
                        .max(MIN_RATING)
                        .min(MAX_RATING);
                    ratings[author].push(rating)
                }
            } else {
                for rating in ratings.iter_mut() {
                    rating.push(MIN_RATING)
                }
            }
        }

        // Adjust ratings non-linearly
        ratings
            .into_iter()
            .map(|probabilities| {
                let probabilities = Self::adjust_probabilities(probabilities);

                let nth = 1.0 / probabilities.len() as f32;
                let probabilities_comp = probabilities.iter().map(|p| 1.0 - *p);

                let p = 1.0 - probabilities_comp.product::<f32>().powf(nth);
                let q = 1.0 - probabilities.iter().product::<f32>().powf(nth);
                let s = (p - q) / (p + q);
                (1.0 + s) / 2.0
            })
            .collect()
    }

    fn adjust_probabilities(mut probabilities: Vec<f32>) -> Vec<f32> {
        if probabilities.len() > 6 {
            probabilities.sort_by(|a, b| a.partial_cmp(b).unwrap());
            // Remove extreme ones if enough
            probabilities = probabilities[2..(probabilities.len() - 2)].to_vec();
            if probabilities.len() > 80 {
                probabilities = [
                    &probabilities[..40],
                    &probabilities[(probabilities.len() - 40)..],
                ]
                .concat();
            }
        }
        probabilities
    }
}

impl Display for BayesianModel {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Bayesian Model with {} authors and {} tokens.",
            self.authors.len(),
            self.total_token_count
        )
    }
}
