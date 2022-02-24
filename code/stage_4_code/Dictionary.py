"""
create word count dict and word-index mapping
"""

class Dictionary():
    frequencies = {}
    dictionary = {'<pad>': 0, '<unk>': 1}
    inverse_dictionary = {}

    def update_frequencies_by_sentence(self, sentence):
        for word in sentence:
            if word not in self.frequencies:
                self.frequencies[word] = 1
            else:
                self.frequencies[word] += 1

    def compute_dictionary(self, max_size: int):
        self.frequencies = sorted(self.frequencies.items(), key=lambda x: x[1], reverse=True)
        for word, freq in self.frequencies:
            self.dictionary[word] = len(self.dictionary)
            if len(self.dictionary) == max_size + 2:
                break
        self.inverse_dictionary = dict((v, k) for k, v in self.dictionary.items())

    def sentence_to_indexes(self, max_len):
        def sec2ind(sentence):
            indexes = []
            if len(sentence) < max_len :
                padding_len = max_len - len(sentence)
                padding = []
                for i in range(0, padding_len):
                    padding.append('<pad>')
                sentence += padding
            else:
                sentence = sentence[0:max_len]
            for word in sentence:
                if word in self.dictionary:
                    indexes.append(self.dictionary[word])
                else:
                    indexes.append(1)
            return indexes
        return sec2ind

    def indexes_to_sentence(self, indexes):
        words = []
        for index in indexes:
            if self.inverse_dictionary.get(index) is not None:
                words.append(self.inverse_dictionary[index])
            else:
                words.append('<unk>')
        return words