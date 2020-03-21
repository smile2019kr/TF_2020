from pandas import read_table
from collections import defaultdict
import numpy as np
import math

class MovieReview:
    def __init__(self, k = 0.5):  #k: 긍정/부정 기준값
        self.k = k
        self.word_probs = []

    def load_corpus(self):
        corpus = read_table('./data/naive_bays.csv', sep=',', encoding='UTF-8')
        return np.array(corpus)

    def is_number(self, param):
        try:
            float(param)
            return True
        except ValueError:
            return False

    def count_words(self, training_set):
        # 학습데이터는 영화리뷰 본문(doc), 평점(point) 으로 구성
        counts = defaultdict(lambda : [0,0])  # [0,0]으로 초기화한 것
        for doc, point in training_set:
            # 영화리뷰가 text일때만 카운팅
            if self.is_number(doc) is False:
                words = doc.split()
                for word in words:
                    counts[word][0 if point > 3.5 else 1] += 1   # 별 셋 반 이면 값이 올라가는 것
        return counts

    def word_probabilities(self, counts, total_class0, total_class1, k):
        # 단어의 빈도수를 [단어, p(w|긍정), p(w|부정)] 형태로 전환
        return [(w,
                 (class0 + k) / (total_class0 + 2 * k),
                 (class1 + k) / (total_class1 + 2 * k))
                for w, (class0, class1) in counts.items()]

    def class0_probability(self, word_probs, doc):
         # 별도 토크나이즈 하지 않고 띄어쓰기만
         docwords = doc.split()
         log_prob_if_class0 = log_prob_if_class1 = 0.0
         # 모든 단어에 대한 반복
         for word, prob_if_class0, prob_if_class1 in word_probs:
             # 만약 리뷰에 word가 나타나면 해당언어가 나올 log에 확률을 더해줌
            if word in docwords:
                log_prob_if_class0 += math.log(prob_if_class0)
                log_prob_if_class1 += math.log(prob_if_class1)
                # 만약 리뷰에 word가 없으면 해당 단어가 나올 log에 확률을 더해줌
                # 나오지않을 확률은 log( 1 - 나올 확률)로 계산
            else:
                log_prob_if_class0 += math.log(1.0 - prob_if_class0)
                log_prob_if_class1 += math.log(1.0 - prob_if_class1)

         prob_if_class0 = math.exp(log_prob_if_class0)
         prob_if_class1 = math.exp(log_prob_if_class1)

         return prob_if_class0 / (prob_if_class0 + prob_if_class1)

    def train(self):
        training_set = self.load_corpus()
        # 범주 0 (긍정), 범주1 (부정)
        num_class0 = len([1 for _, point in training_set if point > 3.5])
        num_class1 = len(training_set) - num_class0
        word_counts = self.count_words(training_set)
        self.word_probs = self.word_probabilities(word_counts, num_class0, num_class1, self.k)

    def classify(self, doc):
        return self.class0_probability(self.word_probs, doc)


if __name__ == '__main__':
    instance = MovieReview()
    instance.train()
    print(instance.classify('끝내준다. 내인생의 최고의 영화'))
    print(instance.classify('재미없어'))

    """결과값
    0.987309502727692
    0.01263560349465949
    """
