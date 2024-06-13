use std::collections::{HashMap, HashSet};
use std::io;

use charabia::Segment;
use text_splitter::TextSplitter;

#[derive(Debug)]
pub enum Language {
    SimplifiedChinese,
    English,
}

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

#[derive(Debug)]
pub struct BayesianModel {
    language: Language,
    authors: Vec<String>,
    token_author_dict: HashMap<String, Vec<u32>>,
    author_token_count: Vec<u32>,
    total_token_count: u32,
}

const MAX_SENTENCE_LENGTH: usize = 96;

const NO_TOKEN_RATING: f32 = 0.4;

const MIN_RATING: f32 = 0.2;

const MAX_RATING: f32 = 0.7;

impl BayesianModel {
    pub fn train(language: Language, dataset: Vec<(String, String)>) -> io::Result<Self> {
        // Find all authors
        let author_dict = dataset
            .iter()
            .map(|(author, _)| author.clone())
            .collect::<HashSet<_>>();
        let authors = author_dict.into_iter().collect::<Vec<_>>();
        let author_count = authors.len();

        // Find all words and their count in each author's texts
        let mut token_author_dict: HashMap<String, Vec<u32>> = HashMap::new();
        for (author, path) in dataset {
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

        Ok(Self {
            language,
            authors,
            author_token_count,
            token_author_dict,
            total_token_count,
        })
    }

    pub fn save(&self) {}

    pub fn load() -> Self {
        todo!()
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
                probabilities = [&probabilities[..40], &probabilities[(probabilities.len() - 40)..]].concat();
            }
        }
        probabilities
    }
}

#[cfg(test)]
mod tests {
    use crate::{BayesianModel, Language};

    #[test]
    fn test_() {
        let dataset = vec![
            (
                "小和尚".to_string(),
                "D:\\BaiduNetdiskDownload\\庙里有个小和尚.txt".to_string(),
            ),
            (
                "御风".to_string(),
                "D:\\BaiduNetdiskDownload\\御风而行.txt".to_string(),
            ),
        ];
        let model = BayesianModel::train(Language::SimplifiedChinese, dataset).unwrap();
        let s = "　　那小婴孩不过三四个月大，白白嫩嫩的可爱样子让讲究万物皆空的和尚们难得地兴奋起来。他偏偏不怕人，黑黝黝的大眼睛骨碌碌地转着，被无数只手轻轻地捏着小脸蛋也不哭，只是咧着小嘴傻呵呵地笑，只有被不小心捏痛了才扁扁嘴，转着小脑袋到处寻找老和尚。";
        // let s = "男人头也不抬地说道。烈日炎炎下，哈利不安地动了动。太热了，哈利想道，特别是这个夏末。他汗湿的袍子罩在身上，厚重且潮湿，在微风下纹丝不动。哈利旁边站着一个戴着眼镜的矮小男人，留着修剪整齐的斑白胡须。出于某种原因男人似乎丝毫不受暑气的影响，他身着的象牙白袍给人带来阴凉的幻觉。";
        let prediction = model.classify_text(s);
        println!("{:#?}", prediction)
    }
}
