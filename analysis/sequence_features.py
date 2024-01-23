import numpy as np
import pandas as pd
import spacy
import datasets
from utils import timestamp
import spacy_alignments as tokenizations


def get_spacy_tag(tok_ix2spacy_ixs, spacy_attribute_list):
    if len(tok_ix2spacy_ixs) == 1:
        return spacy_attribute_list[tok_ix2spacy_ixs[0]]
    elif len(tok_ix2spacy_ixs) == 0:
        return 'NONE'
    else:
        return 'MULTI'


def get_model_labels(model2spacy, spacy_tag):
    return np.array([
        get_spacy_tag(tok_ix2spacy_ixs, spacy_tag)
        for tok_ix2spacy_ixs in model2spacy
    ])


def make_spacy_feature_df(model, token_tensor):

    spacy.prefer_gpu()
    nlp = spacy.load('en_core_web_trf')

    print('Starting spacy processing of dataset...')
    texts = [model.to_string(token_tensor[i, 1:]) for i in range(len(token_tensor))]
    docs = list(nlp.pipe(texts, batch_size=100))
    print('Finished spacy processing of dataset.')

    n, seq_len = token_tensor.shape
    pos_matrix = np.zeros((n, seq_len), dtype='<U5')
    tag_matrix = np.zeros((n, seq_len), dtype='<U5')
    dep_matrix = np.zeros((n, seq_len), dtype='<U10')
    ent_type_matrix = np.zeros((n, seq_len), dtype='<U8')

    morph_number_matrix = np.zeros((n, seq_len), dtype='<U5')
    morph_person_matrix = np.zeros((n, seq_len), dtype='<U5')
    morph_prontype_matrix = np.zeros((n, seq_len), dtype='<U5')
    morph_tense_matrix = np.zeros((n, seq_len), dtype='<U5')
    morph_verbform_matrix = np.zeros((n, seq_len), dtype='<U5')
    morph_aspect_matrix = np.zeros((n, seq_len), dtype='<U5')

    is_alpha_matrix = np.zeros((n, seq_len), dtype='<U5')
    is_stop_matrix = np.zeros((n, seq_len), dtype='<U5')
    is_sent_end_matrix = np.zeros((n, seq_len), dtype='<U5')
    is_sent_begins_matrix = np.zeros((n, seq_len), dtype='<U5')

    for i, doc in enumerate(docs):
        if i % 1000 == 0:
            print(timestamp(), i)

        model_tokens = model.to_str_tokens(token_tensor[i, 1:])
        spacy_tokens = [t.text for t in doc]

        spacy_pos = np.array([t.pos_ for t in doc])
        spacy_tag = np.array([t.tag_ for t in doc])
        spacy_dep = np.array([t.dep_ for t in doc])
        spacy_ent_type = np.array([t.ent_type_ for t in doc])

        spacy_morph_number = np.array(
            [t.morph.to_dict().get('Number', '') for t in doc])
        spacy_morph_person = np.array(
            [t.morph.to_dict().get('Person', '') for t in doc])
        spacy_morph_prontype = np.array(
            [t.morph.to_dict().get('PronType', '') for t in doc])
        spacy_morph_tense = np.array(
            [t.morph.to_dict().get('Tense', '') for t in doc])
        spacy_morph_verbform = np.array(
            [t.morph.to_dict().get('VerbForm', '') for t in doc])
        spacy_morph_aspect = np.array(
            [t.morph.to_dict().get('Aspect', '') for t in doc])

        spacy_is_alpha = np.array([t.is_alpha for t in doc])
        spacy_is_stop = np.array([t.is_stop for t in doc])

        spacy_end_sent = np.array([t.is_sent_end for t in doc])
        spacy_begin_sent = np.array([t.is_sent_start for t in doc])

        gpt2spacy, spacy2gpt = tokenizations.get_alignments(
            model_tokens, spacy_tokens)

        # offset by 1 for BOS
        pos_matrix[i, 1:] = get_model_labels(gpt2spacy, spacy_pos)
        tag_matrix[i, 1:] = get_model_labels(gpt2spacy, spacy_tag)
        dep_matrix[i, 1:] = get_model_labels(gpt2spacy, spacy_dep)
        ent_type_matrix[i, 1:] = get_model_labels(gpt2spacy, spacy_ent_type)

        morph_number_matrix[i, 1:] = get_model_labels(
            gpt2spacy, spacy_morph_number)
        morph_person_matrix[i, 1:] = get_model_labels(
            gpt2spacy, spacy_morph_person)
        morph_prontype_matrix[i, 1:] = get_model_labels(
            gpt2spacy, spacy_morph_prontype)
        morph_tense_matrix[i, 1:] = get_model_labels(
            gpt2spacy, spacy_morph_tense)
        morph_verbform_matrix[i, 1:] = get_model_labels(
            gpt2spacy, spacy_morph_verbform)
        morph_aspect_matrix[i, 1:] = get_model_labels(
            gpt2spacy, spacy_morph_aspect)

        is_alpha_matrix[i, 1:] = get_model_labels(gpt2spacy, spacy_is_alpha)
        is_stop_matrix[i, 1:] = get_model_labels(gpt2spacy, spacy_is_stop)
        # spacy always deleclares end of sequence true
        is_sent_end_matrix[i, 1:-1] = \
            get_model_labels(gpt2spacy, spacy_end_sent)[:-1]
        is_sent_begins_matrix[i, 1:] = get_model_labels(
            gpt2spacy, spacy_begin_sent)

    # see https://universaldependencies.org/u/pos/
    pos_labels = {
        'is_spacy_adj': pos_matrix == 'ADJ',
        'is_spacy_adp': pos_matrix == 'ADP',
        'is_spacy_det': pos_matrix == 'DET',
        'is_spacy_noun': pos_matrix == 'NOUN',
        'is_spacy_pron': pos_matrix == 'PRON',
        'is_spacy_propn': pos_matrix == 'PROPN',
        'is_spacy_punct': pos_matrix == 'PUNCT',
        'is_spacy_verb': pos_matrix == 'VERB',
        'is_spacy_space': pos_matrix == 'SPACE',
        'is_spacy_num': pos_matrix == 'NUM',
        'is_spacy_adv': pos_matrix == 'ADV',
        'is_spacy_aux': pos_matrix == 'AUX',
        'is_spacy_cconj': pos_matrix == 'CCONJ',
        'is_spacy_sconj': pos_matrix == 'SCONJ',
        'is_spacy_intj': pos_matrix == 'INTJ',
        'is_spacy_sym': pos_matrix == 'SYM',
        'is_spacy_part': pos_matrix == 'PART',
        'is_spacy_multi': pos_matrix == 'MULTI',
    }
    dep_label_classes = [
        'dep', 'punct', 'appos', 'pobj', 'prep', 'dobj', 'det', 'nsubj',
        'amod', 'compound', 'ROOT', 'conj', 'advmod', 'nmod', 'cc', 
        'aux', 'nummod', 'advcl', 'attr', 'ccomp', 'poss',
        'npadvmod', 'mark', 'nsubjpass', 'relcl', 'auxpass', 'acl',
        'acomp', 'pcomp', 'xcomp', 'neg', 'meta'
    ]
    dependency_labels = {
        f'is_spacy_{dep}': dep_matrix == dep for dep in dep_label_classes
    }

    morph_labels = {
        'is_singular': morph_number_matrix == 'Sing',
        'is_plural': morph_number_matrix == 'Plur',

        'is_third_person': morph_person_matrix == '3',
        'is_second_person': morph_person_matrix == '2',
        'is_first_person': morph_person_matrix == '1',

        'is_prs_pron': morph_prontype_matrix == 'Prs',
        'is_art_pron': morph_prontype_matrix == 'Art',
        'is_dem_pron': morph_prontype_matrix == 'Dem',
        'is_rel_pron': morph_prontype_matrix == 'Rel',

        'is_past_tense': morph_tense_matrix == 'Past',
        'is_present_tense': morph_tense_matrix == 'Pres',

        'is_fin_verb': morph_verbform_matrix == 'Fin',
        'is_part_verb': morph_verbform_matrix == 'Part',
        'is_inf_verb': morph_verbform_matrix == 'Inf',

        'is_perf_aspect': morph_aspect_matrix == 'Perf',
        'is_prog_aspect': morph_aspect_matrix == 'Prog',
    }

    ent_label_classes = ['PERSON', 'ORG', 'CARDINAL', 'DATE', 'GPE',
        'WORK_OF_', 'PRODUCT', 'LAW', 'PERCENT', 'QUANTITY', 'TIME',
        'NORP', 'FAC', 'ORDINAL', 'MONEY', 'LOC', 'EVENT', 'LANGUAGE']

    ent_labels = {
        f'is_{ent}': ent_type_matrix == ent for ent in ent_label_classes
    }

    misc_labels = {
        'is_stop': is_stop_matrix == 'True',
        'sent_end': is_sent_end_matrix == 'True',
        'sent_begin': is_sent_begins_matrix == 'True',
    }

    label_sets = [
        pos_labels, dependency_labels, morph_labels, ent_labels, misc_labels
    ]
    full_label_dict = {k: v.flatten() for label_set in label_sets
                        for k, v in label_set.items()}

    label_df = pd.DataFrame(full_label_dict)
    return label_df