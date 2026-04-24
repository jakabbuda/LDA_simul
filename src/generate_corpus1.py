import numpy as np

from corpus import SyntheticCorpus
from utils import make_uncorrelated_markov, gini_coeff, markov_stationary

for mw_fr in [1, 5]:
    base_corpus = SyntheticCorpus(n_docs=1000,
                                  num_topics=5,
                                  vocab_size_per_topic=100,
                                  text_len_params={"lam": 100},
                                  generation_mode="stm",
                                  n_groups_prev=2,
                                  prev_covar_imbal=None,
                                  n_groups_cont=2,
                                  cont_covar_imbal=None,
                                  prev_effect_size=1,
                                  cont_effect_size=0.5,
                                  topic_proportions=None,
                                  topic_covar=0,
                                  stopword_ratio=0,
                                  min_word_freq=mw_fr,
                                  topic_signal_boost=5,
                                  overlap_ratio=0,
                                  unstandard_ratio=0,
                                  markov_matrix=None)
    base_corpus.export_for_r(f'simul_data/base_corp/stm_minwordfreq{mw_fr}/')

    base_corpus = SyntheticCorpus(n_docs=1000,
                                  num_topics=5,
                                  vocab_size_per_topic=100,
                                  text_len_params={"lam": 100},
                                  generation_mode='markov',
                                  n_groups_prev=2,
                                  prev_covar_imbal=None,
                                  n_groups_cont=2,
                                  cont_covar_imbal=None,
                                  prev_effect_size=1,
                                  cont_effect_size=0.5,
                                  topic_proportions=None,
                                  topic_covar=0,
                                  stopword_ratio=0,
                                  min_word_freq=mw_fr,
                                  topic_signal_boost=5,
                                  overlap_ratio=0,
                                  unstandard_ratio=0,
                                  markov_matrix=make_uncorrelated_markov(np.array([0.98, 0.98, 0.98, 0.98, 0.98])))  # uncorr, same prop, ~ 50 word
    base_corpus.export_for_r(f'simul_data/base_corp/markov_minwordfreq{mw_fr}/')

    # vary topic proportions
    topic_props_l = [[0.26822778, 0.22352315, 0.19159127, 0.16764236, 0.14901543],  # gini: 12
                     [0.28263795, 0.22611036, 0.1884253, 0.1615074, 0.14131898],  # gini: 14
                     [0.32164661, 0.22974758, 0.17869256, 0.14620301, 0.12371024],  # gini: 19
                     [0.34482759, 0.22988506, 0.17241379, 0.13793103, 0.11494253],  # gini: 22
                     [0.4379562, 0.2189781, 0.1459854, 0.10948905, 0.08759124],  # gini: 32
                     [0.55950266, 0.18650089, 0.11190053, 0.07992895, 0.06216696]]  # gini: 44
    markov_mtr_l = [make_uncorrelated_markov(np.array([0.98, 0.976, 0.972, 0.968, 0.964])),  # gini: 12
                    make_uncorrelated_markov(np.array([0.98, 0.975, 0.97, 0.965, 0.96])),  # gini: 14
                    make_uncorrelated_markov(np.array([0.98, 0.972, 0.964, 0.956, 0.948])),  # gini: 19
                    make_uncorrelated_markov(np.array([0.98, 0.97, 0.96, 0.95, 0.94])),  # gini: 22
                    make_uncorrelated_markov(np.array([0.98, 0.96, 0.94, 0.92, 0.9])),  # gini: 32
                    make_uncorrelated_markov(np.array([0.98, 0.94, 0.90, 0.86, 0.82]))]  # gini: 44

    for tp in topic_props_l:
        corpus = SyntheticCorpus(n_docs=1000,
                                 num_topics=5,
                                 vocab_size_per_topic=100,
                                 text_len_params={"lam": 100},
                                 generation_mode="stm",
                                 n_groups_prev=2,
                                 prev_covar_imbal=None,
                                 n_groups_cont=2,
                                 cont_covar_imbal=None,
                                 prev_effect_size=1,
                                 cont_effect_size=0.5,
                                 topic_proportions=tp,
                                 topic_covar=0,
                                 stopword_ratio=0,
                                 min_word_freq=mw_fr,
                                 topic_signal_boost=5,
                                 overlap_ratio=0,
                                 unstandard_ratio=0,
                                 markov_matrix=None)
        corpus.export_for_r(f'simul_data/topic_prop_{gini_coeff(tp) * 100:.0f}/stm_minwordfreq{mw_fr}/')

    for mrm in markov_mtr_l:
        corpus = SyntheticCorpus(n_docs=1000,
                                 num_topics=5,
                                 vocab_size_per_topic=100,
                                 text_len_params={"lam": 100},
                                 generation_mode='markov',
                                 n_groups_prev=2,
                                 prev_covar_imbal=None,
                                 n_groups_cont=2,
                                 cont_covar_imbal=None,
                                 prev_effect_size=1,
                                 cont_effect_size=0.5,
                                 topic_proportions=None,
                                 topic_covar=0,
                                 stopword_ratio=0,
                                 min_word_freq=mw_fr,
                                 topic_signal_boost=5,
                                 overlap_ratio=0,
                                 unstandard_ratio=0,
                                 markov_matrix=mrm)  # uncorr, same prop, ~ 50 word
        base_corpus.export_for_r(f'simul_data/topic_prop_{gini_coeff(markov_stationary(mrm)) * 100:.0f}/markov_minwordfreq{mw_fr}/')
