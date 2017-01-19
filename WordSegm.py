# encoding: utf8
#from __future__ import unicode_literals
import numpy
import random
import math
import codecs
import collections

class WordSegm():
    MAX_LEN = 5
    AVE_LEN = 2

    def __init__(self):
        self.segm_sentences = []
        self.word_count = collections.defaultdict(int)
        self.num_words = 0
        self.prob_char = collections.defaultdict(int)

    def load_data(self, filename ):
        self.num_words = 0
        self.segm_sentences = []
        self.word_count = collections.defaultdict(int)
        self.sentences = [ line.replace("\n","").replace("\r", "") for line in codecs.open( filename, "r" , "sjis" ).readlines()]
        self.prob_char = collections.defaultdict(int)

        for sentence in self.sentences:
            words = []

            i = 0
            while i<len(sentence):
                # ランダムに切る
                length = random.randint(1,self.MAX_LEN)

                if i+length>=len(sentence):
                    length = len(sentence)-i

                words.append( sentence[i:i+length] )

                i+=length

            self.segm_sentences.append( words )

            # 言語モデルへ追加
            for i,w in enumerate(words):
                self.word_count[w] += 1
                self.num_words += 1

        # 文字が発生する確率計算
        sum = 0
        for sentence in self.sentences:
            for ch in sentence:
                self.prob_char[ch] += 1
                sum += 1

        for ch in self.prob_char.keys():
            self.prob_char[ch] /= float(sum)

    def calc_output_prob(self, w ):
        alpha = 10.0
        prior = 1.0

        # 文字発生確率
        for ch in w:
            prior *= self.prob_char[ch]

        # 長さ
        L = len(w)
        prior *= (self.AVE_LEN**L) * math.exp( -self.AVE_LEN ) / math.factorial(L)

        p = ( self.word_count[w] +  alpha*prior ) / ( self.num_words + alpha )

        return p

    def forward_filtering(self, sentence ):
        T = len(sentence)
        a = numpy.zeros( (len(sentence), self.MAX_LEN+1) )  # 前向き確率

        for t in range(T):
            for k in range(1,self.MAX_LEN+1):
                if t-k+1<0:
                    break

                out_prob = self.calc_output_prob( sentence[t-k+1:t+1] )

                tt = t-k # t-kを終端とする長さkの単語の前の位置
                if tt>=0:
                    # ttまで到達する単語を全て周辺化
                    for kk in range(self.MAX_LEN):
                        a[t,k] += a[tt,kk]
                    a[t,k] *= out_prob
                else:
                    # 最初の単語
                    a[t,k] = out_prob
        return a

    def sample_idx(self, prob ):
        accm_prob = [0,] * len(prob)
        for i in range(len(prob)):
            accm_prob[i] = prob[i] + accm_prob[i-1]

        rnd = random.random() * accm_prob[-1]
        for i in range(len(prob)):
            if rnd <= accm_prob[i]:
                return i

    def backward_sampling(self, a, sentence ):
        T = a.shape[0]
        t = T-1

        words = []

        while True:
            k = self.sample_idx( a[t] )
            w = sentence[t-k+1:t+1]

            words.insert( 0, w )

            t = t-k

            if t<0:
                break

        return words

    def print_result(self):
        print "-------------------------------"
        for words in self.segm_sentences:
            for w in words:
                print w,"|",
            print

    def save_result(self, fname ):
        f = codecs.open( fname ,  "w" , "sjis" )
        for words in self.segm_sentences:
            for w in words:
                f.write( w )
                f.write( " | " )
            f.write("\n")
        f.close()

    def learn(self):
        for i in range(len(self.sentences)):
            sentence = self.sentences[i]
            words = self.segm_sentences[i]

            # 学習データから削除
            for w in words:
                self.word_count[w] -= 1
                self.num_words -= 1

            # foward確率計算
            a = self.forward_filtering( sentence )

            # backward sampling
            words = self.backward_sampling( a, sentence )

            # パラメータ更新
            self.segm_sentences[i] = words
            for w in words:
                self.word_count[w] += 1
                self.num_words += 1

        return

def main():
    segm = WordSegm()
    segm.load_data( "data.txt" )
    segm.print_result()

    for it in range(100):
        print it
        segm.learn()

    segm.print_result()
    segm.save_result("result.txt")
    return

if __name__ == '__main__':
    main()