import torch
import torch.nn as nn
from typing import List, Dict, Tuple

def log_sum_exp_pytorch(vec: torch.Tensor) -> torch.Tensor:
    """
    Calculate the log_sum_exp trick for the tensor.
    :param vec: (batch_size, from_label, to_label).
    :return: (batch_size, to_label)
    """
    maxScores, _ = torch.max(vec, 1)
    maxScores[maxScores == -float("Inf")] = 0
    maxScoresExpanded = maxScores.view(vec.shape[0] ,1 , vec.shape[2]).expand(vec.shape[0], vec.shape[1], vec.shape[2])
    return maxScores + torch.log(torch.sum(torch.exp(vec - maxScoresExpanded), 1))

class LinearCRF(nn.Module):

    def __init__(self, label2idx: Dict[str, int], idx2labels: List[str] = None, START_TAG='START', STOP_TAG='STOP'):
        super(LinearCRF, self).__init__()

        self.label_size = len(label2idx)

        self.label2idx = label2idx
        self.idx2labels = idx2labels
        self.start_idx = self.label2idx[START_TAG]
        self.end_idx = self.label2idx[STOP_TAG]

        init_transition = torch.randn(self.label_size, self.label_size)
        init_transition[:, self.start_idx] = -10000.0
        init_transition[self.end_idx, :] = -10000.0
        self.transition = nn.Parameter(init_transition)
    
    def forward(self, lstm_scores, word_seq_lens, tags, mask):
        """
        Calculate the negative log-likelihood
        :param lstm_scores:
        :param word_seq_lens:
        :param tags:
        :param mask:
        :return:
        """
        all_scores=  self.calculate_all_scores(lstm_scores= lstm_scores)
        unlabeled_score = self.forward_unlabeled(all_scores, word_seq_lens)
        labeled_score = self.forward_labeled(all_scores, word_seq_lens, tags, mask)
        return unlabeled_score - labeled_score

    def forward_unlabeled(self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Calculate the scores with the forward algorithm. Basically calculating the normalization term
        :param all_scores: (batch_size x max_seq_len x num_labels x num_labels) from (lstm scores + transition scores).
        :param word_seq_lens: (batch_size)
        :return: The score for all the possible structures.
        """
        batch_size = all_scores.size(0)
        seq_len = all_scores.size(1)
        dev_num = all_scores.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        alpha = torch.zeros(batch_size, seq_len, self.label_size, device=curr_dev)

        alpha[:, 0, :] = all_scores[:, 0, self.start_idx, :]

        for word_idx in range(1, seq_len):
            # batch_size, self.label_size, self.label_size
            before_log_sum_exp = alpha[:, word_idx-1, :].view(batch_size, self.label_size, 1).expand(batch_size, self.label_size, self.label_size) + all_scores[:, word_idx, :, :]
            alpha[:, word_idx, :] = log_sum_exp_pytorch(before_log_sum_exp)

        # batch_size x label_size
        last_alpha = torch.gather(alpha, 1, word_seq_lens.view(batch_size, 1, 1).expand(batch_size, 1, self.label_size)-1).view(batch_size, self.label_size)
        last_alpha += self.transition[:, self.end_idx].view(1, self.label_size).expand(batch_size, self.label_size)
        last_alpha = log_sum_exp_pytorch(last_alpha.view(batch_size, self.label_size, 1)).view(batch_size)

        # final score for the unlabeled network in this batch, with size: 1
        return torch.sum(last_alpha)
    
    def forward_labeled(self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor, tags: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        '''
        Calculate the scores for the gold instances.
        :param all_scores: (batch, seq_len, label_size, label_size)
        :param word_seq_lens: (batch, seq_len)
        :param tags: (batch, seq_len)
        :param masks: batch, seq_len
        :return: sum of score for the gold sequences Shape: (batch_size)
        '''
        batchSize = all_scores.shape[0]
        sentLength = all_scores.shape[1]

        ## all the scores to current labels: batch, seq_len, all_from_label?
        currentTagScores = torch.gather(all_scores, 3, tags.view(batchSize, sentLength, 1, 1).expand(batchSize, sentLength, self.label_size, 1)).view(batchSize, -1, self.label_size)
        if sentLength != 1:
            tagTransScoresMiddle = torch.gather(currentTagScores[:, 1:, :], 2, tags[:, :sentLength - 1].view(batchSize, sentLength - 1, 1)).view(batchSize, -1)
        tagTransScoresBegin = currentTagScores[:, 0, self.start_idx]
        endTagIds = torch.gather(tags, 1, word_seq_lens.view(batchSize, 1) - 1)
        tagTransScoresEnd = torch.gather(self.transition[:, self.end_idx].view(1, self.label_size).expand(batchSize, self.label_size), 1,  endTagIds).view(batchSize)
        score = torch.sum(tagTransScoresBegin) + torch.sum(tagTransScoresEnd)
        if sentLength != 1:
            score += torch.sum(tagTransScoresMiddle.masked_select(masks.bool()[:, 1:]))
        return score

    def calculate_all_scores(self, lstm_scores: torch.Tensor) -> torch.Tensor:
        """
        Calculate all scores by adding up the transition scores and emissions (from lstm).
        Basically, compute the scores for each edges between labels at adjacent positions.
        This score is later be used for forward-backward inference
        :param lstm_scores: emission scores.
        :return:
        """
        batch_size = lstm_scores.size(0)
        seq_len = lstm_scores.size(1)
        scores = self.transition.view(1, 1, self.label_size, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size) + \
                 lstm_scores.view(batch_size, seq_len, 1, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size)
        return scores

    def decode(self, features, wordSeqLengths) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the batch input
        :param batchInput:
        :return:
        """
        all_scores = self.calculate_all_scores(features)
        bestScores, decodeIdx = self.viterbi_decode(all_scores, wordSeqLengths)
        return bestScores, decodeIdx

    def viterbi_decode(self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use viterbi to decode the instances given the scores and transition parameters
        :param all_scores: (batch_size x max_seq_len x num_labels)
        :param word_seq_lens: (batch_size)
        :return: the best scores as well as the predicted label ids.
               (batch_size) and (batch_size x max_seq_len)
        """
        batchSize = all_scores.shape[0]
        sentLength = all_scores.shape[1]
        dev_num = all_scores.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        scoresRecord = torch.zeros([batchSize, sentLength, self.label_size], device=curr_dev)
        idxRecord = torch.zeros([batchSize, sentLength, self.label_size], dtype=torch.int64, device=curr_dev)
        mask = torch.ones_like(word_seq_lens, dtype=torch.int64, device=curr_dev)
        startIds = torch.full((batchSize, self.label_size), self.start_idx, dtype=torch.int64, device=curr_dev)
        decodeIdx = torch.LongTensor(batchSize, sentLength).to(curr_dev)

        scores = all_scores
        scoresRecord[:, 0, :] = scores[:, 0, self.start_idx, :]  # represent the best current score from the start, is the best
        idxRecord[:,  0, :] = startIds
        for wordIdx in range(1, sentLength):
            ### scoresIdx: batch x from_label x to_label at current index.
            scoresIdx = scoresRecord[:, wordIdx - 1, :].view(batchSize, self.label_size, 1).expand(batchSize, self.label_size,
                                                                                  self.label_size) + scores[:, wordIdx, :, :]
            idxRecord[:, wordIdx, :] = torch.argmax(scoresIdx, 1)  # the best previous label idx to crrent labels
            scoresRecord[:, wordIdx, :] = torch.gather(scoresIdx, 1, idxRecord[:, wordIdx, :].view(batchSize, 1, self.label_size)).view(batchSize, self.label_size)

        lastScores = torch.gather(scoresRecord, 1, word_seq_lens.view(batchSize, 1, 1).expand(batchSize, 1, self.label_size) - 1).view(batchSize, self.label_size)  ##select position
        lastScores += self.transition[:, self.end_idx].view(1, self.label_size).expand(batchSize, self.label_size)
        decodeIdx[:, 0] = torch.argmax(lastScores, 1)
        bestScores = torch.gather(lastScores, 1, decodeIdx[:, 0].view(batchSize, 1))

        for distance2Last in range(sentLength - 1):
            lastNIdxRecord = torch.gather(idxRecord, 1, torch.where(word_seq_lens - distance2Last - 1 > 0, word_seq_lens - distance2Last - 1, mask).view(batchSize, 1, 1).expand(batchSize, 1, self.label_size)).view(batchSize, self.label_size)
            decodeIdx[:, distance2Last + 1] = torch.gather(lastNIdxRecord, 1, decodeIdx[:, distance2Last].view(batchSize, 1)).view(batchSize)

        return bestScores, decodeIdx
        