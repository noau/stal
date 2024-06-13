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
    pub sentences: Vec<(usize, HashMap<String, f32>)>,
    pub sentence_aggregate: HashMap<String, f32>,
}

pub type Bayesian = HashMap<String, (f32, HashMap<String, f32>)>;

#[derive(Debug)]
pub struct BayesianModel {
    language: Language,
    model: Bayesian,
}

impl BayesianModel {
    pub fn train(language: Language, dataset: Vec<(String, String)>) -> io::Result<Self> {
        let dataset_size = dataset.len();
        let authors = dataset
            .iter()
            .map(|(a, _)| a.clone())
            .fold(HashMap::new(), |mut acc, s| {
                *acc.entry(s).or_insert(0) += 1;
                acc
            });
        let mut dict = HashSet::new();
        let mut word_count: HashMap<String, HashMap<String, u32>> = HashMap::new();
        for (author, path) in dataset {
            let text = std::fs::read_to_string(path)?;
            let words = Self::preprocess(&text).into_iter().flat_map(|(_, w)| w);
            let author = word_count.entry(author).or_default();
            for word in words {
                let w = word.to_string();
                dict.insert(w.clone());
                let wc = author.entry(w).or_default();
                *wc += 1;
            }
        }

        let dict_size = dict.len() as f32;
        let model = word_count
            .into_iter()
            .map(|(author, words)| {
                let prior = *authors.get(&author).unwrap() as f32 / dataset_size as f32;
                let word_count = words.values().sum::<u32>() as f32;
                let laplace = word_count + dict_size;
                let likelihood = words
                    .into_iter()
                    .map(|(w, c)| (w, (c as f32 + 1.0) / laplace))
                    .collect::<HashMap<_, _>>();
                (author, (prior, likelihood))
            })
            .collect::<HashMap<_, (_, HashMap<_, _>)>>();

        Ok(Self { language, model })
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

    pub fn predicate(&self, text: &str) -> Predication {
        let sentences = Self::preprocess(text)
            .iter()
            .map(|(index, words)| {
                (
                    *index,
                    self.model
                        .iter()
                        .map(|(a, (l, wc))| {
                            (
                                a.clone(),
                                *l * words
                                    .iter()
                                    .map(|w| wc.get(&w.to_string()).unwrap_or(&0.1))
                                    .product::<f32>(),
                            )
                        })
                        .collect::<HashMap<_, _>>(),
                )
            })
            .collect::<Vec<_>>();

        let sentence_aggregate =
            sentences
                .iter()
                .map(|(_, map)| map)
                .fold(HashMap::new(), |mut acc, map| {
                    for (key, value) in map {
                        *acc.entry(key.clone()).or_insert(0.0) += value;
                    }
                    acc
                });

        Predication {
            sentences,
            sentence_aggregate,
        }
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
        println!("{:#?}", model);
        // let s = "　　那小婴孩不过三四个月大，白白嫩嫩的可爱样子让讲究万物皆空的和尚们难得地兴奋起来。他偏偏不怕人，黑黝黝的大眼睛骨碌碌地转着，被无数只手轻轻地捏着小脸蛋也不哭，只是咧着小嘴傻呵呵地笑，只有被不小心捏痛了才扁扁嘴，转着小脑袋到处寻找老和尚。";
        let s = "男人头也不抬地说道。烈日炎炎下，哈利不安地动了动。太热了，哈利想道，特别是这个夏末。他汗湿的袍子罩在身上，厚重且潮湿，在微风下纹丝不动。哈利旁边站着一个戴着眼镜的矮小男人，留着修剪整齐的斑白胡须。出于某种原因男人似乎丝毫不受暑气的影响，他身着的象牙白袍给人带来阴凉的幻觉。";
        let prediction = model.predicate(s);
        println!("{:#?}", prediction)
    }
}
