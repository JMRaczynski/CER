# CER (Coherent Explainable Recommender)

Augmented PETER+ [[1]] architecture for providing explanations which are more coherent than plain PETER+ ones.
Code is based on original PETER/PETER+ code, available at [[2]].

Links for datasets to download can be found on repository with PETER code [[2]].

## Usage

Install all the packages listed in [requirements.txt](requirements.txt) file. All experiments were run
with Python 3.8.

GPU is not required but strongly recommended.

Below, there is an example of how to run CER on Unix based system:

```commandline
python -u main.py \
--data_path ../datasets/TripAdvisor/reviews.pickle \
--index_dir ../datasets/TripAdvisor/1/ \
--cuda \
--checkpoint ./tripadvisorf/ \
--peter_mask \
--use_feature \
--cer >> tripadvisorf.log
```

If you want to run PETER+ architecture instead, just do not add `--cer` argument, like this:

```commandline
python -u main.py \
--data_path ../datasets/TripAdvisor/reviews.pickle \
--index_dir ../datasets/TripAdvisor/1/ \
--cuda \
--checkpoint ./tripadvisorf/ \
--peter_mask \
--use_feature >> tripadvisorf.log
```

In both options, such a run should create three new files:
- binary file with serialized trained model
- text file with output for the test set. In this file, each example is represented with 4 lines:
  - first line consists of ground truth explanation and rating, delimited with single space
  - second line is a context predicted by PETER/CER
  - third line consists of PETER/CER explanation and rating, also delimited with single space
  - fourth line is empty, to delimit examples
- text file with logs generated during training and evaluation, called `tripadvisorf.log` in above case

Repository also contains `coherence_automatic_evaluation.py` script, which allows to reproduce
proposed automatic coherence evaluation approach. Various settings for automatic evaluation script
can be set with constants defined in lines 145-185.

## If you use the code, please cite:
```
@inproceedings{ECAI23-CER,
	title={The Problem of Coherence in Natural Language Explanations of Recommendations},
	author={Raczyński, Jakub and Lango, Mateusz and Stefanowski, Jerzy},
	booktitle={ECAI},
	year={2023}
}
```
```
@inproceedings{ACL21-PETER,
	title={Personalized Transformer for Explainable Recommendation},
	author={Li, Lei and Zhang, Yongfeng and Chen, Li},
	booktitle={ACL},
	year={2021}
}
```

[1]: https://lileipisces.github.io/files/ACL21-PETER-paper.pdf
[2]: https://github.com/lileipisces/PETER