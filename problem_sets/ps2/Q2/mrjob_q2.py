from mrjob.job import MRJob
from mrjob.step import MRStep
from heapq import nlargest
from operator import itemgetter
import re

WORD_RE = re.compile(r"[\w']+")

class MRMostUsedWord(MRJob):
    def mapper_get_words(self, _, row):
        data = row.split('\t')
        for word in WORD_RE.findall(data[2]):
            yield (word.lower(), 1)

    def combiner_count_words(self, word, counts):
        yield (word, sum(counts))

    def reducer_count_words(self, word, counts):
        yield None, (sum(counts), word)

    def reducer_find_max_word(self, _, word_count_pairs):
        yield None, sorted(word_count_pairs, reverse=True)[:10]

    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_words,
                   combiner=self.combiner_count_words,
                   reducer=self.reducer_count_words),
            MRStep(reducer=self.reducer_find_max_word)
        ]

if __name__=='__main__':
    MRMostUsedWord.run()