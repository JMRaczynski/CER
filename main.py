import argparse
import math
import os
from typing import Tuple

import torch
import torch.nn as nn

from module import PETER
from cer import CER
from utils import rouge_score, bleu_score, DataLoader, Batchify, now_time, ids2tokens, unique_sentence_percent, \
    root_mean_square_error, mean_absolute_error, feature_detect, feature_matching_ratio, feature_coverage_ratio, \
    feature_diversity


parser = argparse.ArgumentParser(description='PErsonalized Transformer for Explainable Recommendation (PETER)')
parser.add_argument('--data_path', type=str, default=None,
                    help='path for loading the pickle data')
parser.add_argument('--index_dir', type=str, default=None,
                    help='load indexes')
parser.add_argument('--emsize', type=int, default=512,
                    help='size of embeddings')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the transformer')
parser.add_argument('--nhid', type=int, default=2048,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--lr', type=float, default=1.0,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log_interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--checkpoint', type=str, default='./peter/',
                    help='directory to save the final model')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--vocab_size', type=int, default=20000,
                    help='keep the most frequent words in the dict')
parser.add_argument('--endure_times', type=int, default=5,
                    help='the maximum endure times of loss increasing on validation')
parser.add_argument('--rating_reg', type=float, default=0.1,
                    help='regularization on recommendation task')
parser.add_argument('--context_reg', type=float, default=1.0,
                    help='regularization on context prediction task')
parser.add_argument('--text_reg', type=float, default=1.0,
                    help='regularization on text generation task')
parser.add_argument('--peter_mask', action='store_true',
                    help='True to use peter mask; Otherwise left-to-right mask')
parser.add_argument('--use_feature', action='store_true',
                    help='False: no feature; True: use the feature')
parser.add_argument('--words', type=int, default=15,
                    help='number of words to generate for each sample')
parser.add_argument('--cer', action='store_true',
                    help='True to use CER with additional recommender; Otherwise PETER architecture is used')
args = parser.parse_args()

if args.data_path is None:
    parser.error('--data_path should be provided for loading data')
if args.index_dir is None:
    parser.error('--index_dir should be provided for loading data splits')

print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')
device = torch.device('cuda' if args.cuda else 'cpu')

if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)
model_file_name = 'cer' if args.cer else 'peter'
if args.cer:
    model_file_name += 'r'
model_path = os.path.join(args.checkpoint, model_file_name + '.pt')
prediction_path = os.path.join(args.checkpoint, args.outf)

###############################################################################
# Load data
###############################################################################

print(now_time() + 'Loading data')
corpus = DataLoader(args.data_path, args.index_dir, args.vocab_size)
word2idx = corpus.word_dict.word2idx
idx2word = corpus.word_dict.idx2word
feature_set = corpus.feature_set
train_data = Batchify(corpus.train, word2idx, args.words, args.batch_size, shuffle=True)
val_data = Batchify(corpus.valid, word2idx, args.words, args.batch_size)
test_data = Batchify(corpus.test, word2idx, args.words, args.batch_size)

###############################################################################
# Build the model
###############################################################################
src_len = 2
if args.use_feature:
    src_len += train_data.feature.size(1)
tgt_len = args.words + 1  # added <bos> or <eos>
ntokens = len(corpus.word_dict)
nuser = len(corpus.user_dict)
nitem = len(corpus.item_dict)
pad_idx = word2idx['<pad>']
if args.cer:
    model = CER(args.peter_mask, src_len, tgt_len, pad_idx, nuser, nitem, ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
else:
    model = PETER(args.peter_mask, src_len, tgt_len, pad_idx, nuser, nitem, ntokens, args.emsize, args.nhead,
                  args.nhid, args.nlayers, args.dropout).to(device)
text_criterion = nn.NLLLoss(ignore_index=pad_idx)  # ignore the padding when computing loss
rating_criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.25)

###############################################################################
# Training code
###############################################################################


def predict(log_context_dis, topk):
    word_prob = log_context_dis.exp()  # (batch_size, ntoken)
    if topk == 1:
        context = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1)
    else:
        context = torch.topk(word_prob, topk, 1)[1]  # (batch_size, topk)
    return context  # (batch_size, topk)


def train(data):
    # Turn on training mode which enables dropout.
    model.train()
    context_loss = 0.
    text_loss = 0.
    rating_loss = 0.
    total_sample = 0
    while True:
        user, item, rating, seq, feature = data.next_batch()  # (batch_size, seq_len), data.step += 1
        batch_size = user.size(0)
        user = user.to(device)  # (batch_size,)
        item = item.to(device)
        rating = rating.to(device)
        seq = seq.t().to(device)  # (tgt_len + 1, batch_size)
        feature = feature.t().to(device)  # (1, batch_size)
        if args.use_feature:
            text = torch.cat([feature, seq[:-1]], 0)  # (src_len + tgt_len - 2, batch_size)
        else:
            text = seq[:-1]  # (src_len + tgt_len - 2, batch_size)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()
        log_word_prob, log_context_dis, ratings_p, _ = model(user, item, text, rating)  # (tgt_len, batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
        context_dis = log_context_dis.unsqueeze(0).repeat((tgt_len - 1, 1, 1))  # (batch_size, ntoken) -> (tgt_len - 1, batch_size, ntoken)
        c_loss = text_criterion(context_dis.view(-1, ntokens), seq[1:-1].reshape((-1,)))
        r_loss = rating_criterion(ratings_p[0], rating)
        if args.cer:
            r_loss += rating_criterion(ratings_p[1], ratings_p[0])
        t_loss = text_criterion(log_word_prob.view(-1, ntokens), seq[1:].reshape((-1,)))
        loss = args.text_reg * t_loss + args.context_reg * c_loss + args.rating_reg * r_loss
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        context_loss += batch_size * c_loss.item()
        text_loss += batch_size * t_loss.item()
        rating_loss += batch_size * r_loss.item()
        total_sample += batch_size

        if data.step % args.log_interval == 0 or data.step == data.total_step:
            cur_c_loss = context_loss / total_sample
            cur_t_loss = text_loss / total_sample
            cur_r_loss = rating_loss / total_sample
            print(now_time() + 'context ppl {:4.4f} | text ppl {:4.4f} | rating loss {:4.4f} | {:5d}/{:5d} batches'.format(
                math.exp(cur_c_loss), math.exp(cur_t_loss), cur_r_loss, data.step, data.total_step))
            context_loss = 0.
            text_loss = 0.
            rating_loss = 0.
            total_sample = 0
        if data.step == data.total_step:
            break


def evaluate(data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    context_loss = 0.
    text_loss = 0.
    rating_loss = 0.
    total_sample = 0
    with torch.no_grad():
        while True:
            user, item, rating, seq, feature = data.next_batch()  # (batch_size, seq_len), data.step += 1
            batch_size = user.size(0)
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            rating = rating.to(device)
            seq = seq.t().to(device)  # (tgt_len + 1, batch_size)
            feature = feature.t().to(device)  # (1, batch_size)
            if args.use_feature:
                text = torch.cat([feature, seq[:-1]], 0)  # (src_len + tgt_len - 2, batch_size)
            else:
                text = seq[:-1]  # (src_len + tgt_len - 2, batch_size)
            log_word_prob, log_context_dis, ratings_p, _ = model(user, item, text, None)  # (tgt_len, batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
            context_dis = log_context_dis.unsqueeze(0).repeat((tgt_len - 1, 1, 1))  # (batch_size, ntoken) -> (tgt_len - 1, batch_size, ntoken)
            c_loss = text_criterion(context_dis.view(-1, ntokens), seq[1:-1].reshape((-1,)))
            r_loss = rating_criterion(ratings_p[0], rating)
            if args.cer:
                r_loss += rating_criterion(ratings_p[1], ratings_p[0])
            t_loss = text_criterion(log_word_prob.view(-1, ntokens), seq[1:].reshape((-1,)))

            context_loss += batch_size * c_loss.item()
            text_loss += batch_size * t_loss.item()
            rating_loss += batch_size * r_loss.item()
            total_sample += batch_size

            if data.step == data.total_step:
                break
    return context_loss / total_sample, text_loss / total_sample, rating_loss / total_sample


def generate_greedy(model, model_input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    user, item, text = model_input
    for idx in range(args.words):  # produce a word at each step
        log_word_prob, _, _, _ = model(user, item, text, None, False, False, False)  # (batch_size, ntoken)
        word_prob = log_word_prob.exp()  # (batch_size, ntoken)
        word_idx = torch.argmax(word_prob, dim=1)  # (batch_size,), pick the one with the largest probability
        text = torch.cat([text, word_idx.unsqueeze(0)], 0)  # (len++, batch_size)
    return text


def generate_with_beam_search(model, model_input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], beam_size=5):  # all words represented as indices
    user, item, text = model_input
    repeated_user, repeated_item = user.repeat_interleave(beam_size), item.repeat_interleave(beam_size)
    batch_size = text.shape[-1]
    log_word_prob, _, _, _ = model(user, item, text, None, False, False, False)
    beam_log_probas, top_candidate_words = torch.topk(log_word_prob, beam_size, dim=1)
    sentences = torch.cat([text.repeat_interleave(beam_size, dim=1), top_candidate_words.reshape(1, -1)], 0)
    beam_log_probas = beam_log_probas.unsqueeze(-1)
    for _ in range(args.words - 1):
        log_word_prob, _, _, _ = model(repeated_user, repeated_item, sentences, None, False, False, False)  # (batch_size * beam_size, ntoken)
        top_candidate_log_probas, top_candidate_words = torch.topk(log_word_prob, beam_size, dim=1, sorted=False)  # (batch_size * beam_size, beam_size)
        beam_log_probas = beam_log_probas + top_candidate_log_probas.reshape(batch_size, beam_size, beam_size)
        beam_log_probas, probas_idx = torch.topk(beam_log_probas.reshape(batch_size, -1), beam_size, dim=1, sorted=False)
        beam_log_probas = beam_log_probas.unsqueeze(-1)
        new_words = torch.gather(top_candidate_words.reshape(batch_size, -1), 1, probas_idx)
        sentences_idx = (probas_idx / beam_size).long()  # calculating sentence idx basing on candidate word idx
        sentences = sentences.T.reshape(batch_size, beam_size, -1)[torch.arange(0, batch_size).reshape(-1, 1), sentences_idx, :]
        sentences = torch.cat([sentences.reshape(batch_size * beam_size, -1).T, new_words.reshape(1, -1)], 0)
    best_sentences_idx = beam_log_probas.squeeze().argmax(1)
    return sentences.T.reshape(batch_size, beam_size, -1)[range(batch_size), best_sentences_idx, :].T


def generate(data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    idss_predict = []
    context_predict = []
    rating_predict = []
    with torch.no_grad():
        while True:
            user, item, rating, seq, feature = data.next_batch()
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            bos = seq[:, 0].unsqueeze(0).to(device)  # (1, batch_size)
            feature = feature.t().to(device)  # (1, batch_size)
            if args.use_feature:
                text = torch.cat([feature, bos], 0)  # (src_len - 1, batch_size)
            else:
                text = bos  # (src_len - 1, batch_size)
            start_idx = text.size(0)
            log_word_prob, log_context_dis, ratings_p, _ = model(user, item, text, None, False)  # (batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
            rating_predict += ratings_p[0].tolist()
            context_predict += predict(log_context_dis, topk=args.words).tolist()  # (batch_size, words)
            text = generate_greedy(model, (user, item, text))
            ids = text[start_idx:].t().tolist()  # (batch_size, seq_len)
            idss_predict.extend(ids)

            if data.step == data.total_step:
                break

    # rating
    predicted_rating = [(r, p) for (r, p) in zip(data.rating.tolist(), rating_predict)]
    RMSE = root_mean_square_error(predicted_rating, corpus.max_rating, corpus.min_rating)
    print(now_time() + 'RMSE {:7.4f}'.format(RMSE))
    MAE = mean_absolute_error(predicted_rating, corpus.max_rating, corpus.min_rating)
    print(now_time() + 'MAE {:7.4f}'.format(MAE))
    # text
    tokens_test = [ids2tokens(ids[1:], word2idx, idx2word) for ids in data.seq.tolist()]
    tokens_predict = [ids2tokens(ids, word2idx, idx2word) for ids in idss_predict]
    BLEU1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
    print(now_time() + 'BLEU-1 {:7.4f}'.format(BLEU1))
    BLEU4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
    print(now_time() + 'BLEU-4 {:7.4f}'.format(BLEU4))
    USR, USN = unique_sentence_percent( tokens_predict)
    print(now_time() + 'USR {:7.4f} | USN {:7}'.format(USR, USN))
    feature_batch = feature_detect(tokens_predict, feature_set)
    DIV = feature_diversity(feature_batch)  # time-consuming
    print(now_time() + 'DIV {:7.4f}'.format(DIV))
    FCR = feature_coverage_ratio(feature_batch, feature_set)
    print(now_time() + 'FCR {:7.4f}'.format(FCR))
    feature_test = [idx2word[i] for i in data.feature.squeeze(1).tolist()]  # ids to words
    FMR = feature_matching_ratio(feature_batch, feature_test)
    print(now_time() + 'FMR {:7.4f}'.format(FMR))
    text_test = [' '.join(tokens) for tokens in tokens_test]
    text_predict = [' '.join(tokens) for tokens in tokens_predict]
    tokens_context = [' '.join([idx2word[i] for i in ids]) for ids in context_predict]
    ROUGE = rouge_score(text_test, text_predict)  # a dictionary
    for (k, v) in ROUGE.items():
        print(now_time() + '{} {:7.4f}'.format(k, v))
    text_out = ''
    for (real, real_rating, ctx, fake, fake_rating) in zip(text_test, data.rating.tolist(), tokens_context, text_predict, rating_predict):
        text_out += '{} {}\n{}\n{} {}\n\n'.format(real, real_rating, ctx, fake, fake_rating)
    return text_out


# Loop over epochs.

# with open(model_path, 'rb') as f:
#     model = torch.load(f).to(device)

best_val_loss = float('inf')
endure_count = 0
for epoch in range(1, args.epochs + 1):
    print(now_time() + 'epoch {}'.format(epoch))
    train(train_data)
    val_c_loss, val_t_loss, val_r_loss = evaluate(val_data)
    if args.rating_reg == 0:
        val_loss = val_t_loss
    else:
        val_loss = val_t_loss + val_r_loss
    print(now_time() + 'context ppl {:4.4f} | text ppl {:4.4f} | rating loss {:4.4f} | valid loss {:4.4f} on validation'.format(
        math.exp(val_c_loss), math.exp(val_t_loss), val_r_loss, val_loss))
    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        with open(model_path, 'wb') as f:
            torch.save(model, f)
    else:
        endure_count += 1
        print(now_time() + 'Endured {} time(s)'.format(endure_count))
        if endure_count == args.endure_times:
            print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
            break
        # Anneal the learning rate if no improvement has been seen in the validation dataset.
        scheduler.step()
        print(now_time() + 'Learning rate set to {:2.8f}'.format(scheduler.get_last_lr()[0]))

# Load the best saved model.
with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)

# Run on test data.
test_c_loss, test_t_loss, test_r_loss = evaluate(test_data)
print('=' * 89)
print(now_time() + 'context ppl {:4.4f} | text ppl {:4.4f} | rating loss {:4.4f} on test | End of training'.format(
    math.exp(test_c_loss), math.exp(test_t_loss), test_r_loss))

print(now_time() + 'Generating text')
text_o = generate(test_data)
with open(prediction_path, 'w', encoding='utf-8') as f:
    f.write(text_o)
print(now_time() + 'Generated text saved to ({})'.format(prediction_path))
