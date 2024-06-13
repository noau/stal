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
    dict: HashMap<String, Vec<u32>>,
    author_words: Vec<u32>,
    word_count: u32,
}

impl BayesianModel {
    pub fn train(language: Language, dataset: Vec<(String, String)>) -> io::Result<Self> {
        let author_dict = dataset
            .iter()
            .map(|(a, _)| a.clone())
            .collect::<HashSet<_>>();
        let authors = author_dict.into_iter().collect::<Vec<_>>();

        let mut dict: HashMap<String, Vec<u32>> = HashMap::new();
        for (author, path) in dataset {
            let author_index = authors.iter().position(|s| author.eq(s)).unwrap();

            let text = std::fs::read_to_string(path)?;
            let words = Self::preprocess(&text).into_iter().flat_map(|(_, w)| w);
            for word in words {
                let w = word.to_string();
                let w = dict.entry(w).or_insert(vec![0; authors.len()]);
                w[author_index] += 1;
            }
        }

        let author_words = dict.iter().fold(vec![0; authors.len()], |aw, (_, wc)| {
            aw.iter()
                .zip(wc.iter())
                .map(|(a, b)| *a + *b)
                .collect::<Vec<_>>()
        });
        let word_count = author_words.iter().sum::<u32>();

        Ok(Self {
            language,
            authors,
            author_words,
            dict,
            word_count,
        })
    }

    pub fn save(&self) {}

    pub fn load() -> Self {
        todo!()
    }

    pub fn preprocess(string: &str) -> Vec<(usize, Vec<&str>)> {
        let ts = TextSplitter::new(96);
        ts.chunk_indices(string)
            .map(|(index, sentence)| (index, sentence.segment_str().collect::<Vec<_>>()))
            .collect::<Vec<_>>()
    }

    pub fn classify(&self, text: &str) -> Classification {
        let transform = |v: Vec<f32>| {
            v.into_iter()
                .enumerate()
                .map(|(a, p)| (self.authors[a].clone(), p))
                .collect()
        };
        let Predication {
            sentences_predicate,
            total_predicate,
        } = self.predicate(text);
        let sentences_classification = sentences_predicate
            .into_iter()
            .map(|(s, p)| (s, transform(p)))
            .collect::<Vec<_>>();
        let total_classification = transform(total_predicate);
        Classification {
            sentences_classification,
            total_classification,
        }
    }

    fn predicate(&self, text: &str) -> Predication {
        let sentences = Self::preprocess(text);
        let sentences_predicate = sentences
            .iter()
            .map(|(s, sentence)| (*s, self.predicate_sentence(sentence)))
            .collect::<Vec<_>>();
        let total_predicate = sentences_predicate.iter().map(|(_, s)| s).fold(
            vec![0.0; self.authors.len()],
            |t, s| {
                t.iter()
                    .zip(s.iter())
                    .map(|(&a, &b)| a + b)
                    .collect::<Vec<_>>()
            },
        );
        Predication {
            sentences_predicate,
            total_predicate,
        }
    }

    fn predicate_sentence(&self, sentence: &Vec<&str>) -> Vec<f32> {
        let mut ratings = vec![vec![]; self.authors.len()];

        for word in sentence {
            let w = word.to_string();
            if let Some(word_count) = self.dict.get(&w) {
                let count = word_count.iter().sum::<u32>();
                for a in 0..self.authors.len() {
                    let ac = word_count[a];
                    if ac == 0 {
                        ratings[a].push(0.2);
                    }

                    let this_prob = ac as f32 / self.author_words[a] as f32;
                    let other_prob =
                        (count - ac) as f32 / (self.word_count - self.author_words[a]) as f32;
                    let rating = (this_prob / (this_prob + other_prob)).max(0.2).min(0.7);
                    ratings[a].push(rating)
                }
            } else {
                for rating in ratings.iter_mut() {
                    rating.push(0.2)
                }
            }
        }

        ratings
            .into_iter()
            .map(|mut probs| {
                if probs.len() > 6 {
                    probs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    probs = probs[2..(probs.len() - 2)].to_vec();
                    if probs.len() > 80 {
                        probs = [&probs[..40], &probs[(probs.len() - 40)..]].concat();
                    }
                }

                let nth = 1.0 / probs.len() as f32;
                let probs_comp = probs.iter().map(|p| 1.0 - *p);

                let p = 1.0 - probs_comp.product::<f32>().powf(nth);
                let q = 1.0 - probs.iter().product::<f32>().powf(nth);
                let s = (p - q) / (p + q);
                (1.0 + s) / 2.0
            })
            .collect()
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
        // println!("{:#?}", model);
        // let s = "　　那小婴孩不过三四个月大，白白嫩嫩的可爱样子让讲究万物皆空的和尚们难得地兴奋起来。他偏偏不怕人，黑黝黝的大眼睛骨碌碌地转着，被无数只手轻轻地捏着小脸蛋也不哭，只是咧着小嘴傻呵呵地笑，只有被不小心捏痛了才扁扁嘴，转着小脑袋到处寻找老和尚。";
        let s = "男人头也不抬地说道。烈日炎炎下，哈利不安地动了动。太热了，哈利想道，特别是这个夏末。他汗湿的袍子罩在身上，厚重且潮湿，在微风下纹丝不动。哈利旁边站着一个戴着眼镜的矮小男人，留着修剪整齐的斑白胡须。出于某种原因男人似乎丝毫不受暑气的影响，他身着的象牙白袍给人带来阴凉的幻觉。";
        let prediction = model.classify(s);
        println!("{:#?}", prediction)
    }
}
