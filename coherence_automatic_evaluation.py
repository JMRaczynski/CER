import time
from typing import Tuple, List

import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
import torch
from transformers import BertModel, BertTokenizer


class CoherencePredictionModel(torch.nn.Module):

    def __init__(self):
        super(CoherencePredictionModel, self).__init__()
        self.dense_stack = torch.nn.Sequential(
            torch.nn.Linear(768, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 2)
        )

    def forward(self, input):
        return self.dense_stack(input)


def get_bert_sentence_representation(sentences: list, bert, tokenizer):
    return bert(**tokenizer(sentences, return_tensors="pt", padding=True).to(DEVICE)).pooler_output.detach()


def load_data(sentence_path: str, rating_path: str) -> Tuple[List[str], torch.Tensor]:
    sentences = read_file_with_data(sentence_path)
    ratings = torch.Tensor([int(rating) for rating in read_file_with_data(rating_path)])
    return sentences, ratings


def load_coherence_data(path: str) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    lines = read_file_with_data(path)
    split_lines = [line.split("\t") for line in lines]
    samples = [(sentence.split(" "), int(label)) for _, sentence, label in split_lines]
    samples = [(" ".join(sentence_with_rating[:-1]), round(float(sentence_with_rating[-1])), label) for sentence_with_rating, label in samples]
    sentences, ratings, labels = [], [], []
    for sentence, rating, coherence_label in samples:
        sentences.append(sentence)
        ratings.append(rating)
        labels.append(coherence_label)
    return sentences, torch.Tensor(ratings), torch.Tensor(labels).type(torch.LongTensor)


def load_peter_output(path: str, mod=2) -> Tuple[List[str], List[int]]:
    with open(path, "r") as f:
        lines = f.readlines()
    split_lines = [line.split(" ") for i, line in enumerate(lines) if i % 4 == mod]
    predicted_ratings = [round(float(line[-1])) for line in split_lines]
    return [PROMPT_TEXTS[rating] + '"' + " ".join(line[:-1]) + '"' for rating, line in zip(predicted_ratings, split_lines)], predicted_ratings


def read_file_with_data(path: str) -> List[str]:
    with open(path, "r") as f:
        lines = f.readlines()
    return lines


def cross_validate(X, y, model_type, metrics, epoch_num=50):
    split_generator = StratifiedKFold(n_splits=10, shuffle=True)
    splits = split_generator.split(X.cpu(), y.cpu())
    fold_results = []
    for i, (train_ind, test_ind) in enumerate(splits):
        print(f"\n{i + 1}-fold:")
        X_train, X_test, y_train, y_test = X[train_ind], X[test_ind], y[train_ind], y[test_ind]
        model, optimizer, loss_function = init_model(model_type)
        train(model, optimizer, loss_function, X_train, y_train, epoch_num)
        evaluation_results = evaluate(model, metrics, X_test, y_test)
        print(f"Test set variance: {torch.var(y_test.type(torch.FloatTensor))}")
        for metric, results in zip(metrics, evaluation_results):
            print(f"{metric}: {results}")
        fold_results.append(evaluation_results)
    averaged_results_over_fold = torch.mean(torch.Tensor(fold_results), 0)
    print("\nResults averaged over folds:")
    print(f"Variance: {torch.var(y.type(torch.FloatTensor))}")
    for metric, results in zip(metrics, averaged_results_over_fold):
        print(f"{metric}: {results}")
    print()


def init_model(model_type):
    model = model_type().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=L2_REGULARIZATION_WEIGHT)
    loss_function = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
    return model, optimizer, loss_function.to(DEVICE)


def train(model, optimizer, loss_function, X: torch.Tensor, y: torch.Tensor, n_epochs=50, verbose=False):
    batch_num = len(y) / BATCH_SIZE
    for epoch in range(1, n_epochs + 1):
        loss_sum = torch.tensor(0.)
        shuffled_indices = torch.randperm(len(y))
        X, y = X[shuffled_indices], y[shuffled_indices]
        for i in range(0, len(y), BATCH_SIZE):
            pred = model(X[i:i + BATCH_SIZE]).squeeze()
            loss = loss_function(pred, y[i:i + BATCH_SIZE])
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            optimizer.zero_grad()
            loss_sum += loss.cpu()
        if verbose or epoch == n_epochs:
            print(f"Epoch {epoch} loss: {loss_sum / batch_num}")


def evaluate(model, metrics, X, y):
    y_pred = infer(model, X)
    if set(metrics) == set(CLASSIFICATION_METRICS):
        y_pred = torch.argmax(y_pred, dim=1)
    # print(y_pred)
    return [metric(y.cpu(), y_pred.cpu()) for metric in metrics]


def evaluate_on_full_dataset(model, bert, tokenizer, data_path, mod=2):
    test_sentences, predicted_ratings = load_peter_output(data_path, mod)
    number_of_coherent_samples = 0
    coherence_predictions = []
    for i in range(0, len(test_sentences), INFERENCE_BATCH_SIZE):
        inference_results = infer(model, get_bert_sentence_representation(test_sentences[i:i + INFERENCE_BATCH_SIZE], bert, tokenizer))
        predicted_coherencies = torch.argmax(inference_results, dim=1)
        number_of_coherent_samples += torch.sum(predicted_coherencies)
        coherence_predictions += list(predicted_coherencies.cpu().numpy())
        # print("Predictions:", infer(model, get_bert_sentence_representation(test_sentences[i:i + INFERENCE_BATCH_SIZE], bert, tokenizer)))
    print(compute_per_class_coherencies(coherence_predictions, predicted_ratings))
    return number_of_coherent_samples.item() / len(test_sentences), coherence_predictions


def compute_per_class_coherencies(coherence_predictions: list, predicted_ratings: list) -> dict:
    results_per_class = dict.fromkeys(range(1, 6))
    for key in results_per_class:
        boolean_mask = np.array(predicted_ratings) == key
        results_per_class[key] = round(100 * np.sum(np.array(coherence_predictions)[boolean_mask]) / np.sum(boolean_mask), 2)
    for i in results_per_class.values():
        print(i)
    return results_per_class


def infer(model, X):
    with torch.no_grad():
        prediction = model(X).squeeze()
    return prediction

PRETRAINED_MODEL_NAME = "bert-base-uncased"
DEVICE = torch.device("cuda")
BATCH_SIZE = 10
INFERENCE_BATCH_SIZE = 64

# DATASET = "AmazonMovies"
# CLASS_WEIGHTS = torch.Tensor([1.8, 1.])
# L2_REGULARIZATION_WEIGHT = 0.05
# EPOCH_NUM = 100

# DATASET = "TripAdvisor"
# CLASS_WEIGHTS = torch.Tensor([2., 1.])
# L2_REGULARIZATION_WEIGHT = 0.1
# EPOCH_NUM = 150

DATASET = "Yelp"
CLASS_WEIGHTS = torch.Tensor([2.5, 1.])
L2_REGULARIZATION_WEIGHT = 0.05
EPOCH_NUM = 150


SENTENCE_FILE_PATH = f"../models/{DATASET}/gt explanations sample.txt"
GROUND_TRUTH_RATING_FILE_PATH = f"../models/{DATASET}/gt sample labels.txt"
HUMAN_RATING_FILE_PATH = f"../models/{DATASET}/gt sample human labels.txt"
COHERENCE_FILE_PATH = f"../models/{DATASET}/gt sample coherence labels.txt"
PETER_OUTPUT_PATH = f"../models/{DATASET}/generated{DATASET.lower()}.txt"
CER_OUTPUT_PATH = f"../models/{DATASET}/generated_cer_yelp.txt"
COHERENCE_MODEL_DIR = f"coherence_models/{DATASET}/100_samples"
LOAD_COHERENCE_MODEL = True

CLASSIFICATION_METRICS = [accuracy_score, recall_score, f1_score, precision_score]

PROMPT_TEXTS = {
    1: "An example of a very negative review is ",
    2: "An example of a slightly negative review is ",
    3: "An example of a neutral or mixed review is ",
    4: "An example of a slightly positive review is ",
    5: "An example of a very positive review is "
}


def main():
    print(f"Dataset: {DATASET}")
    sentences, ratings = load_data(SENTENCE_FILE_PATH, GROUND_TRUTH_RATING_FILE_PATH)
    sentences = [PROMPT_TEXTS[int(rating.item())] + '"' + sentence + '"' for sentence, rating in zip(sentences, ratings)]
    coherence_labels = torch.Tensor([int(is_coherent) for is_coherent
                                     in read_file_with_data(COHERENCE_FILE_PATH)]).type(torch.LongTensor).to(DEVICE)

    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    bert = BertModel.from_pretrained(PRETRAINED_MODEL_NAME).to(DEVICE)

    model_input_features = get_bert_sentence_representation(sentences, bert, tokenizer)

    results_peter, results_cer = [], []
    start = time.time()
    for i in range(10):
        model, optimizer, loss_function = init_model(CoherencePredictionModel)

        # cross_validate(model_input_features, coherence_labels, CoherencePredictionModel, CLASSIFICATION_METRICS, EPOCH_NUM)

        checkpoint_path = f"{COHERENCE_MODEL_DIR}/model{i + 1}.pt"
        if LOAD_COHERENCE_MODEL:
            print(f"{i + 1} loaded")
            model.load_state_dict(torch.load(checkpoint_path))
        else:
            print(f"{i + 1} trained")
            train(model, optimizer, loss_function, model_input_features, coherence_labels, EPOCH_NUM)
            torch.save(model.state_dict(), checkpoint_path)
        model.eval()
        # print(evaluate(model, CLASSIFICATION_METRICS, model_input_features, coherence_labels))
        results, predictions = evaluate_on_full_dataset(model, bert, tokenizer, PETER_OUTPUT_PATH)
        results_peter.append(results)
        results_modification, predictions_modification = evaluate_on_full_dataset(model, bert, tokenizer, CER_OUTPUT_PATH)
        results_cer.append(results_modification)
    print(f"Elapsed time: {time.time() - start}")
    print("PETER coherence:", [round(i * 100, 2) for i in results_peter], np.mean(results_peter))
    print("CER coherence:", [round(i * 100, 2) for i in results_cer], np.mean(results_cer))


if __name__ == "__main__":
    main()
