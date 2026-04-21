import numpy as np
import pandas as pd
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
                 stopword_boost: float = 10.0,
                 topic_signal_boost: float = 6.0,
                 overlap_ratio: float = 0.0,
                 unstandard_ratio: float = 0.0,
                 max_variants: int = 3,
                 zipf_s: float = 1.1,
                 markov_matrix: Union[np.ndarray, None] = None):
        """

        :param n_docs:
        :param num_topics:
        :param vocab_size_per_topic:
        :param text_len_dist:
        :param text_len_params:
        :param generation_mode:
        :param n_groups_prev:
        :param prev_covar_imbal:
        :param n_groups_cont:
        :param cont_covar_imbal:
        :param prev_effect_size:
        :param cont_effect_size:
        :param topic_correlation:
        :param stopword_ratio:
        :aparm n_stopwords:
        :param stopword_boost:
        :param topic_signal_boost:
        :param overlap_ratio:
        :param unstandard_ratio:
        :aparm max_variants:
        :param zipf_s:
        :param markov_matrix:
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
                raise ValueError(f"prev_covar_imbal should have one valu for each prevalence covariate group")
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
        self.unstd_ratio = unstandard_ratio
        self.vocab_size_per_topic = vocab_size_per_topic if isinstance(vocab_size_per_topic, list) else [vocab_size_per_topic] * (num_topics if isinstance(num_topics, int) else sum(num_topics))
        self.zipf_s = zipf_s
        self.n_groups_prev = n_groups_prev
        self.n_groups_cont = n_groups_cont
        # Signal control
        self.stopword_boost = stopword_boost
        self.topic_signal_boost = topic_signal_boost

        # Vocabulary & distribution
        self.full_vocab, self.topic_to_symbols = self._build_vocab(overlap_ratio, n_stopwords, max_variants)
        self.v_size = len(self.full_vocab)
        self.word_to_idx = {w: i for i, w in enumerate(self.full_vocab)}

        max_t = max(self.num_topics_list)
        self._init_beta_components(cont_effect_size)

        # Prevalence (gamma) and topic covariance (Sigma)
        self.gamma = np.random.normal(0, prev_effect_size, (max_t, self.n_groups_prev))
        if isinstance(topic_correlation, np.ndarray):
            self.sigma = topic_correlation
        else:
            self.sigma = np.eye(max_t) * 1.0
            if max_t > 1:
                self.sigma[self.sigma == 0] = topic_correlation

        # Markov setup
        if markov_matrix is not None:
            self.markov_matrices = [markov_matrix] * self.n_groups_prev
        else:
            self.markov_matrices = [np.random.dirichlet([0.5] * max_t, max_t) for _ in range(self.n_groups_prev)]

        # execution
        self.documents, self.metadata, self.ground_truth_theta = [], [], []
        self._synthesize()

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
                t1, t2 = np.random.randint(0, max_t, 2)
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

        stopwords = [f"stop{i}" for i in range(n_stopwords)]
        all_symbols.extend(stopwords)
        self.stopword_indices = []

        full_vocab = sorted(list(set(all_symbols)))
        self.stopword_indices = [full_vocab.index(s) for s in stopwords]
        return full_vocab, final_map

    def _init_beta_components(self, c_eff):
        ranks = np.arange(1, self.v_size + 1)
        self.m = np.log((1 / (ranks ** self.zipf_s)) / (1 / (ranks ** self.zipf_s)).sum())
        # Stopword head noise
        for idx in self.stopword_indices:
            self.m[idx] += self.stopword_boost

        max_t = max(self.num_topics_list)
        self.kappa_k = np.zeros((max_t, self.v_size))
        for t in range(max_t):
            for word in self.topic_to_symbols[t]:
                if word in self.word_to_idx:
                    self.kappa_k[t, self.word_to_idx[word]] = self.topic_signal_boost

        self.kappa_kg = np.random.normal(0, c_eff, (max_t, self.n_groups_cont, self.v_size))

    def _get_beta(self, t_idx, g_idx):
        return softmax(self.m + self.kappa_k[t_idx] + self.kappa_kg[t_idx, g_idx % self.n_groups_cont])

    def _generate_doc(self, n_t, g_p, g_c, length, M):
        if self.mode == 'stm':
            meta_p = np.zeros(self.n_groups_prev)
            meta_p[g_p % self.n_groups_prev] = 1
            # Slicing gamma to match n_t
            mean_vec = np.dot(self.gamma[:n_t, :], meta_p)
            cov_mat = self.sigma[:n_t, :n_t]
            eta = np.random.multivariate_normal(mean_vec, cov_mat)
            theta = softmax(eta)
            z = np.random.choice(n_t, size=length, p=theta)
        else:
            z = [np.random.choice(n_t)]
            for _ in range(length - 1):
                p = M[z[-1]][:n_t] / M[z[-1]][:n_t].sum()
                z.append(np.random.choice(n_t, p=p))
            theta = np.bincount(z, minlength=n_t) / length

        doc = [self.full_vocab[np.random.choice(self.v_size, p=self._get_beta(zi, g_c))] for zi in z]
        return " ".join(doc), theta

    def _synthesize(self):
        for i in range(self.n_subcorpora):
            n_t = self.num_topics_list[i]
            for _ in range(self.n_docs[i]):
                gp = np.random.choice(range(self.gamma.shape[1]), p=self.prev_covar_imbal)
                gc = np.random.choice(range(self.kappa_kg.shape[1]), p=self.cont_covar_imbal)
                n_words = self.text_len_dist(**self.len_params[i])
                M = self.markov_matrices[gp % len(self.markov_matrices)]
                txt, th = self._generate_doc(n_t, gp, gc, n_words, M)
                self.documents.append(txt)
                self.metadata.append({'subcorpus': i, 'prev_covar': gp, 'content_covar': gc})
                self.ground_truth_theta.append(th)

    def get_gold_standard(self):
        """
        Returns the true beta and theta for evaluation
        """
        max_t = max(self.num_topics_list)
        true_betas = {}
        for g in range(self.n_groups_cont):
            true_betas[f"group_{g}"] = np.array([self._get_beta(t, g) for t in range(max_t)])

        return {
            "vocab": self.full_vocab,
            "topic_to_symbols": self.topic_to_symbols,
            "stopword_list": [self.full_vocab[i] for i in self.stopword_indices],
            "true_betas": true_betas,
            "true_thetas": self.ground_truth_theta,
            "metadata": self.metadata
        }

    def export_for_r(self, path_prefix="corpus"):
        """Exports DTM and metadata for R consumption."""
        from sklearn.feature_extraction.text import CountVectorizer
        # fixed vocabulary to ensure R and Python are aligned
        vectorizer = CountVectorizer(vocabulary=self.full_vocab, token_pattern=r"(?u)\b\w+\b")
        dtm = vectorizer.transform(self.documents)

        # Save DTM as CSV
        pd.DataFrame(dtm.toarray(), columns=self.full_vocab).to_csv(f"{path_prefix}_dtm.csv", index=False)
        pd.DataFrame(self.metadata).to_csv(f"{path_prefix}_meta.csv", index=False)
        print(f"Exported to {path_prefix}_dtm.csv and {path_prefix}_meta.csv")

    def get_data(self):
        return pd.DataFrame(self.metadata).assign(text=self.documents)


if __name__ == "__main__":
    # simple LDA (no noise, no covariates, no correlation)
    lda_toy = SyntheticCorpus(
        n_docs=11, num_topics=3,
        text_len_params={"lam": 7},
        stopword_ratio=0.0,
        prev_effect_size=0, cont_effect_size=0, topic_correlation=0,
        overlap_ratio=0.1, unstandard_ratio=0.2
    )
    print("--- Toy 1: Simple LDA ---")
    print(lda_toy.get_data())

    # STM (correlation + covariates)
    stm_toy = SyntheticCorpus(
        n_docs=11, num_topics=3,
        text_len_params={"lam": 7},
        prev_effect_size=2.0,  # Strong prevalence effect
        cont_effect_size=1.5,  # Strong content effect
        topic_correlation=0.5,  # Correlated topics
        n_groups_prev=2, n_groups_cont=2
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
        text_len_params={"lam": 15},
        generation_mode='markov',
        markov_matrix=corr_m
    )
    print("\n--- Toy 3: Markov ---")
    print(markov_toy.get_data())

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
