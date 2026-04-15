from typing import Union, List

import numpy as np
import string


class SyntheticCorpus:

    def __init__(self,
                 size: int,
                 num_topic: int,
                 noise_ratio: float = 0,
                 topic_word_num_mean: Union[float, int, None] = None,
                 topic_word_num_sd: Union[float, int, None] = None,
                 topic_word_num_list: Union[int, List[int], None] = None,
                 text_len_mean: Union[float, int, None] = None,
                 text_len_sd: Union[float, int, None] = None,
                 topic_transition_matrix: Union[np.matrix, None] = None
                 ):
        """

        :param size: size of the corpus
        :param num_topic: number of topics
            topic words will be marked with letters a - y (z is reserved for noise, as a simultion of stopwords)
            if num_topic > 25, then topic names will be repeted letter: string.ascii_lowercase[topic_num % 25] * (topic_num // 25)
        :param noise_ratio: float between 0-1: proportion of noise ("stop words"),
            will be added after the texts are created: noise_ratio * lenn text stowords will be added to random positions
            if noise_ratio = 0: mimics data completly cleaned from stopwords, amd without any non topic specific words
        :param topic_word_num_mean:
        :param topic_word_num_sd:
        :param topic_word_num_list:
        :param text_len_mean:
        :param text_len_sd:
        :param topic_transition_matrix:
        """
        self.size = size
        self.topic_num = num_topic
        assert 0 <= noise_ratio <= 1, f"noise_ratio should be between 0 and 1, but it is {noise_ratio}"
        self.noise_ratio = noise_ratio

        if isinstance(topic_word_num_list, list):
            assert len(topic_word_num_list) == num_topic

        self.synthetize_corpus()

    def __repr__(self):
        pass
        # return f"{type(self).__name__} with width {self.char_width} & height {self.char_height}"

    def __str__(self):
        pass
        # return f'{type(self).__name__}(component_list={self.component_list})'

    def synthetize_corpus(self):
        corpus = np.random.random(self.size)

        return corpus

