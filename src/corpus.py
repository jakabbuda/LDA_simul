from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import string
from scipy.special import softmax
from typing import List, Union, Callable


class SyntheticCorpus:
    def __init__(self,
                 n_docs: Union[int, List[int]],
                 num_topics: Union[int, List[int]],
                 vocab_size_per_topic: Union[int, List[int]] = 500,
                 text_len_dist: Callable = np.random.poisson,
                 text_len_params: Union[None, dict, List[dict]] = None,
                 generation_mode: str = 'stm',
                 n_groups_prev: int = 2,
                 prev_covar_imbal: Union[None, list] = None,
                 n_groups_cont: int = 2,
                 cont_covar_imbal: Union[None, list] = None,
                 prev_effect_size: float = 0.0,
                 cont_effect_size: float = 0.0,
                 topic_correlation: Union[float, np.ndarray] = 0.0,
                 stopword_ratio: float = 0.0,
                 n_stopwords: int = 50,
                 min_word_freq: int = 1,
                 topic_signal_boost: float = 6.0,
                 overlap_ratio: float = 0.0,
                 unstandard_ratio: float = 0.0,
                 max_variants: int = 3,
                 zipf_s: float = 1.1,
                 markov_matrix: Union[np.ndarray, None] = None):
        """
        Synthetic corpus generation

        :param n_docs: number of documents
        :param num_topics: number of topics
            for composite corpora with different generative process for different topics this is a list
        :param vocab_size_per_topic: number of words in topics, can be same for all, or different
        :param text_len_dist: distribution of document length in words
            can be list for composite corpus
        :param text_len_params: parameter(s) for text_len_dist distribution
            can be list for composite corpus
        :param generation_mode: 'stm' for stm or lda
                                OR
                                'markov'
        :param n_groups_prev: number of categorical prevalence covariate values
        :param prev_covar_imbal: parameter to set imbalance of prevalence covariate classes
            should be None for balanced classes or list summing to 1 of length n_groups_prev for setting proportions
        :param n_groups_cont: number of categorical content covariate values
        :param cont_covar_imbal: parameter to set imbalance of content covariate classes
            should be None for balanced classes or list summing to 1 of length n_groups_cont for setting
        :param prev_effect_size: prevalence covariate effect size
        :param cont_effect_size: content covariate effect size
        :param topic_correlation: topic correlation - can be defined by one number (same pairwise corr)
            or with a covariance matrix
        :param stopword_ratio: ratio of stopwords (stop1, stop2, ...)
        :param n_stopwords: number of different stopwords
        :param min_word_freq: minimum frequency of words in corpus
            simulating infrequent word removal during preprocessing
        :param topic_signal_boost: defines topic words boosted probability
            larger numbers results in cleaner topics
        :param overlap_ratio: ratio of topic tokens that are associated with multiple topics
        :param unstandard_ratio: ratio of tokens that are not standard forms (signald with _v#)
        :param max_variants: number of different forms for unstandard words
        :param zipf_s: zipf distribution exponent param
        :param markov_matrix: markov transition between topics for mode 'markov'
        """

        if text_len_params is None:
            text_len_params = {"lam": 100}
        self.n_subcorpora = len(n_docs) if isinstance(n_docs, list) else 1
        self.n_docs = self._to_list(n_docs)
        self.num_topics_list = self._to_list(num_topics)
        self.len_params = self._to_list(text_len_params)
        self.text_len_dist = text_len_dist

        if prev_covar_imbal is not None:
            if len(prev_covar_imbal) != n_groups_prev:
                raise ValueError(f"prev_covar_imbal should have one value for each prevalence covariate group")
            self.prev_covar_imbal = prev_covar_imbal
        else:
            self.prev_covar_imbal = [1/n_groups_prev] * n_groups_prev

        if cont_covar_imbal is not None:
            if len(cont_covar_imbal) != n_groups_cont:
                raise ValueError(f"cont_covar_imbal should have one value for each prevalence covariate group")
            self.cont_covar_imbal = cont_covar_imbal
        else:
            self.cont_covar_imbal = [1/n_groups_cont] * n_groups_cont

        self.mode = generation_mode
        self.stopword_ratio = stopword_ratio
        self.min_word_freq = min_word_freq
        self.unstd_ratio = unstandard_ratio
        self.zipf_s = zipf_s
        self.n_groups_prev = n_groups_prev
        self.n_groups_cont = n_groups_cont
        self.topic_signal_boost = topic_signal_boost

        # Vocabulary & distribution
        self.vocab_size_per_topic = vocab_size_per_topic if isinstance(vocab_size_per_topic, list) else [vocab_size_per_topic] * (num_topics if isinstance(num_topics, int) else sum(num_topics))
        self.full_vocab, self.topic_to_symbols, self.stopwords = self._build_vocab(overlap_ratio, n_stopwords, max_variants)
        self.v_size = len(self.full_vocab)
        self.word_to_idx = {w: i for i, w in enumerate(self.full_vocab)}

        self._init_generative_params(cont_effect_size, prev_effect_size, topic_correlation, markov_matrix)

        # execution
        self.documents, self.metadata, self.ground_truth_theta = [], [], []
        self._synthesize()

        # removing rare words
        if self.min_word_freq > 1:
            self._remove_rare_words()

    def _to_list(self, p):
        return p if isinstance(p, list) else [p] * self.n_subcorpora

    def _build_vocab(self, overlap_ratio, n_stopwords, max_variants):
        max_t = max(self.num_topics_list)
        topic_map = {t: [f"{string.ascii_lowercase[t % 26] * (t // 26 + 1)}{i}" for i in
                         range(self.vocab_size_per_topic[t])] for t in
                     range(max_t)}

        # Shared words across topics
        if overlap_ratio > 0 and max_t > 1:
            n_ov = int(sum(self.vocab_size_per_topic) * overlap_ratio)
            for overl in range(n_ov):
                t1, t2 = np.random.choice(range(max_t), 2, replace=False)
                i_t1 = np.random.randint(0, len(topic_map[t1]))
                i_t2 = np.random.randint(0, len(topic_map[t2]))
                shared = f"{topic_map[t1][i_t1]}_{topic_map[t2][i_t2]}"
                topic_map[t1][i_t1] = topic_map[t2][i_t2] = shared

        # Expand for unstandardization (not sufficient lemmatization / stemming)
        final_map = {t: [] for t in range(max_t)}
        all_symbols = []
        for t, words in topic_map.items():
            for w in words:
                if np.random.random() < self.unstd_ratio:
                    v_count = np.random.randint(2, max_variants + 1)
                    variants = [f"{w}_v{v}" for v in range(v_count)]
                    final_map[t].extend(variants)
                    all_symbols.extend(variants)
                else:
                    final_map[t].append(w)
                    all_symbols.append(w)

        # stopwords
        stopwords = []
        if self.stopword_ratio > 0:
            stopwords = [f"stop{i}" for i in range(n_stopwords)]
            all_symbols.extend(stopwords)

        full_vocab = sorted(list(set(all_symbols)))
        return full_vocab, final_map, stopwords

    def _init_generative_params(self, c_eff, p_eff, correlation, markov_m):
        ranks = np.arange(1, self.v_size + 1)
        # m: background freq - defined according zipfian disrib
        self.m = np.log((1 / (ranks ** self.zipf_s)) / (1 / (ranks ** self.zipf_s)).sum())

        max_t = max(self.num_topics_list)
        self.kappa_k = np.zeros((max_t, self.v_size))  # topic effect
        for t in range(max_t):
            for word in self.topic_to_symbols[t]:
                # if word in self.word_to_idx:
                self.kappa_k[t, self.word_to_idx[word]] = np.random.normal(self.topic_signal_boost,
                                                                           self.topic_signal_boost/5)

        self.kappa_kg = np.random.normal(0, c_eff, (max_t, self.n_groups_cont, self.v_size))  # conetnt cov effect
        self.gamma = np.random.normal(0, p_eff, (max_t, self.n_groups_prev))  # topic prevalence effect
        if isinstance(correlation, np.ndarray):
            self.sigma = correlation
        else:
            self.sigma = np.eye(max_t)
            if max_t > 1:
                self.sigma[self.sigma == 0] = correlation

        if markov_m is not None:
            self.markov_matrices = [markov_m] * self.n_groups_prev
        else:
            self.markov_matrices = [np.random.dirichlet([0.5] * max_t, max_t) for _ in range(self.n_groups_prev)]

    def _get_beta(self, t_idx, g_idx):
        # log_evidence(word v, topic k): score_v=m_v + kappa_k,v + kappa_g,v
        log_beta = self.m + self.kappa_k[t_idx] + self.kappa_kg[t_idx, g_idx % self.n_groups_cont]
        # Mask out stopwords for the topical distribution
        if self.stopwords:
            stopword_indices = [self.word_to_idx[s] for s in self.stopwords]
            log_beta[stopword_indices] = -1e10
        return softmax(log_beta)

    def _generate_doc(self, n_t, g_p, g_c, length, M):
        if self.mode == 'stm':
            meta_p = np.zeros(self.n_groups_prev)
            meta_p[g_p] = 1
            # Slicing gamma to match n_t
            mean_vec = np.dot(self.gamma[:n_t, :], meta_p)  # expected topic distr in group
            cov_mat = self.sigma[:n_t, :n_t]
            eta = np.random.multivariate_normal(mean_vec, cov_mat)  # actual topic distr (log)
            theta = softmax(eta)
            doc_topic_list = np.random.choice(n_t, size=length, p=theta)
        else:  # markov
            doc_topic_list = [np.random.choice(n_t)]
            for _ in range(length - 1):
                p = M[doc_topic_list[-1]][:n_t] / M[doc_topic_list[-1]][:n_t].sum()
                doc_topic_list.append(np.random.choice(n_t, p=p))
            theta = np.bincount(doc_topic_list, minlength=n_t) / length

        doc = []
        stopword_indices = [self.word_to_idx[s] for s in self.stopwords]
        stopword_probs = []
        if self.stopword_ratio > 0:
            stopword_probs = softmax(self.m[stopword_indices])

        for ti in doc_topic_list:
            if self.stopword_ratio > 0 and np.random.random() < self.stopword_ratio:
                doc.append(np.random.choice(self.stopwords, p=stopword_probs))
            else:
                doc.append(np.random.choice(self.full_vocab, p=self._get_beta(ti, g_c)))
        return doc, theta

    def _synthesize(self):
        for i in range(self.n_subcorpora):
            n_t = self.num_topics_list[i]
            for _ in range(self.n_docs[i]):
                gp = np.random.choice(range(self.gamma.shape[1]), p=self.prev_covar_imbal)
                gc = np.random.choice(range(self.kappa_kg.shape[1]), p=self.cont_covar_imbal)
                n_words = self.text_len_dist(**self.len_params[i])
                M = self.markov_matrices[gp]
                txt, th = self._generate_doc(n_t, gp, gc, n_words, M)
                self.documents.append(txt)
                self.metadata.append({'subcorpus': i, 'prev_covar': gp, 'content_covar': gc})
                self.ground_truth_theta.append(th)

    def _remove_rare_words(self):
        """remove words appearing less than min_word_freq"""
        all_tokens = [t for doc in self.documents for t in doc]
        counts = Counter(all_tokens)
        to_keep = {w for w, c in counts.items() if c >= self.min_word_freq}
        self.documents = [[t for t in doc if t in to_keep] for doc in self.documents]

    def get_gold_standard(self):
        """
        returns the true beta and theta for evaluation
        """
        max_t = max(self.num_topics_list)
        true_betas = {}
        for g in range(self.n_groups_cont):
            true_betas[f"group_{g}"] = np.array([self._get_beta(t, g) for t in range(max_t)])
        actual_vocab = sorted(list(set([t for doc in self.documents for t in doc])))
        return {
            "vocab": actual_vocab,
            "topic_to_symbols": self.topic_to_symbols,
            "stopword_list": self.stopwords,
            "true_betas": true_betas,
            "true_thetas": self.ground_truth_theta,
            "metadata": self.metadata
        }

    def export_for_r(self, path_prefix="corpus"):
        docs_as_strings = [" ".join(doc) for doc in self.documents]
        vectorizer = CountVectorizer(vocabulary=self.full_vocab, token_pattern=r"(?u)\b\w+\b")
        dtm = vectorizer.transform(docs_as_strings)
        # Save DTM as CSV
        pd.DataFrame(dtm.toarray(), columns=self.full_vocab).to_csv(f"{path_prefix}_dtm.csv", index=False)
        pd.DataFrame(self.metadata).to_csv(f"{path_prefix}_meta.csv", index=False)
        print(f"Exported to {path_prefix}_dtm.csv and {path_prefix}_meta.csv")

    def get_data(self):
        docs_as_strings = [" ".join(doc) for doc in self.documents]
        return pd.DataFrame(self.metadata).assign(text=docs_as_strings)


if __name__ == "__main__":
    # simple LDA (no noise, no covariates, no correlation)
    lda_toy = SyntheticCorpus(
        n_docs=11, num_topics=3,
        text_len_params={"lam": 7},
        stopword_ratio=0.5,
        prev_effect_size=0, cont_effect_size=0, topic_correlation=0,
        overlap_ratio=0.2, unstandard_ratio=0.4,
        vocab_size_per_topic=[10, 7, 3], n_stopwords=5,
        min_word_freq=2
    )
    print("--- Toy 1: Simple LDA ---")
    print(lda_toy.get_data())
    # for k, v in lda_toy.get_gold_standard().items():
    #     print(f"  {'-'*15} {k} {'-'*15}")
    #     print(v)

    # # STM (correlation + covariates)
    stm_toy = SyntheticCorpus(
        n_docs=11, num_topics=3,
        text_len_params={"lam": 7},
        prev_effect_size=2.0,  # Strong prevalence effect
        cont_effect_size=1.5,  # Strong content effect
        topic_correlation=0.5,  # Correlated topics
        n_groups_prev=2, n_groups_cont=2,
        min_word_freq=2
    )
    print("\n--- Toy 2: STM ---")
    print(stm_toy.get_data())

    # Markov (Transition Matrix provided)
    corr_m = np.array([
        [0.6, 0.3, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8]
    ])
    markov_toy = SyntheticCorpus(
        n_docs=11, num_topics=3,
        text_len_params={"lam": 7},
        generation_mode='markov',
        markov_matrix=corr_m,
        vocab_size_per_topic=[10, 7, 3], n_stopwords=5,
        stopword_ratio=0.2, unstandard_ratio=0.2, overlap_ratio=0.2
    )
    print("\n--- Toy 3: Markov ---")
    print(markov_toy.get_data())
    # for k, v in markov_toy.get_gold_standard().items():
    #     print(f"  {'-'*15} {k} {'-'*15}")
    #     print(v)
    #
    # Composite corpus (Bimodal/Unbalanced)
    # 5 short docs with 2 topics, 6 long docs with 4 topics
    composite_toy = SyntheticCorpus(
        n_docs=[5, 6],
        num_topics=[2, 4],
        text_len_params=[{"lam": 5}, {"lam": 50}],
        stopword_ratio=0.1  # Adding some noise to see the 'stop' symbols
    )
    print("\n--- Toy 4: Composite ---")
    data = composite_toy.get_data()
    print(data.groupby('subcorpus').agg({'text': lambda x: x.iloc[0][:50], 'prev_covar': 'count'}))
