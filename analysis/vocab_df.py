from functools import partial
import numpy as np
import pandas as pd
import torch


# symbolic token, numeric token, alpha token


TYPE_FEATURES = {
    'all_white_space': lambda x: x.strip() == '',
    'all_caps': lambda x: x.strip().isupper() and len(x.strip()) > 1,
    'all_lower': lambda x: x.strip().islower(),
    'all_alpha': lambda x: x.strip().isalpha(),
    'all_numeric': lambda x: x.strip().isdecimal(),
    'all_symbolic': lambda x: any(not c.isalnum() for c in x.strip())
}

SYMBOL_FEATURES = {
    'contains_period': lambda x: '.' in x,
    'contains_comma': lambda x: ',' in x,
    'contains_exclamation': lambda x: '!' in x,
    'contains_question': lambda x: '?' in x,
    'contains_semicolon': lambda x: ';' in x,
    'contains_colon': lambda x: ':' in x,
    'contains_dash': lambda x: '-' in x,
    'contains_slash': lambda x: '/' in x,
    'contains_backslash': lambda x: '\\' in x,
    'contains_underscore': lambda x: '_' in x,
    'contains_carrot': lambda x: '^' in x,
    'contains_apostrophe': lambda x: '\'' in x,
    'contains_quotation': lambda x: '\"' in x,
    'contains_close_paren': lambda x: ')' in x,
    'contains_open_paren': lambda x: '(' in x,
    'contains_close_bracket': lambda x: ']' in x,
    'contains_open_bracket': lambda x: '[' in x,
    'contains_close_brace': lambda x: '}' in x,
    'contains_open_brace': lambda x: '{' in x,
    'contains_closing_punc': lambda x: any([c in x for c in [')', ']', '}', '>']]),
    'contains_opening_punc': lambda x: any([c in x for c in ['(', '[', '{', '<']]),
    'contains_math_symbol': lambda x: any([c in x for c in ['+', '-', '*', '/', '=', '<', '>']]),
    'contains_currency': lambda x: any([c in x for c in ['$', '€', '£', '¥']]),
    'contains_dollar': lambda x: '$' in x,
    'contains_double_dollar': lambda x: '$$' in x,
}

NUMBER_WORDS = set([
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
    'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty',
    'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand', 'million',
    'billion', 'trillion', 'quadrillion', 'quintillion', 'sextillion',
    'hundreds', 'thousands', 'millions', 'billions', 'trillions', 'dozen', 'dozens'
])

IS_QUANITY_WORD = None  # TODO

NUMERIC_FEATURES = {
    'contains_digit': lambda x: any([c.isdigit() for c in x]),
    'all_digits': lambda x: all([c.isdigit() for c in x.strip()]),
    'is_number_word': lambda x: x.strip() in NUMBER_WORDS,
    'is_year': lambda x: x.strip().isdecimal() and int(x.strip()) > 1490 and int(x.strip()) < 2200,
    'integral_value': lambda x: int(x.strip()) if x.strip().isdecimal() else np.nan,
    'is_one_digit': lambda x: x.strip().isdecimal() and len(x.strip()) == 1,
    'is_two_digit': lambda x: x.strip().isdecimal() and len(x.strip()) == 2,
    'is_three_digit': lambda x: x.strip().isdecimal() and len(x.strip()) == 3,
    'is_four_digit': lambda x: x.strip().isdecimal() and len(x.strip()) == 4,
    'is_lt_31': lambda x: x.strip().isdecimal() and int(x.strip()) <= 31,
}

PRONOUN_FEATURES = {
    'is_male_pronoun': lambda x: x.strip().lower() in set(['he', 'him', 'his', 'himself']),
    'is_female_pronoun': lambda x: x.strip().lower() in set(['she', 'her', 'hers', 'herself']),
    'is_neutral_pronoun': lambda x: x.strip().lower() in set(['they', 'them', 'their', 'theirs', 'themself', 'themselves']),
    'is_first_person_pronoun': lambda x: x.strip().lower() in set(['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves']),
    'is_second_person_pronoun': lambda x: x.strip().lower() in set(['you', 'your', 'yours', 'yourself', 'yourselves']),
    'is_third_person_pronoun': lambda x: x.strip().lower() in set(['he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'they', 'them', 'their', 'theirs', 'themself', 'themselves']),
    'is_plural_pronoun': lambda x: x.strip().lower() in set(['we', 'us', 'our', 'ours', 'ourselves', 'they', 'them', 'their', 'theirs', 'themself', 'themselves']),
    'is_singular_pronoun': lambda x: x.strip().lower() in set(['i', 'me', 'my', 'mine', 'myself', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself']),
    'is_subject_pronoun': lambda x: x.strip().lower() in set(['i', 'we', 'you', 'he', 'she', 'it', 'they']),
    'is_object_pronoun': lambda x: x.strip().lower() in set(['me', 'us', 'you', 'him', 'her', 'it', 'them']),
    'is_possessive_pronoun': lambda x: x.strip().lower() in set(['my', 'mine', 'our', 'ours', 'your', 'yours', 'his', 'her', 'hers', 'its', 'their', 'theirs']),
    'is_reflexive_pronoun': lambda x: x.strip().lower() in set(['myself', 'ourselves', 'yourself', 'yourselves', 'himself', 'herself', 'itself', 'themselves']),
    'is_personal_pronoun': lambda x: x.strip().lower() in set(['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'they', 'them', 'their', 'theirs', 'themself', 'themselves']),
    'is_indefinite_pronoun': lambda x: x.strip().lower() in set(['anybody', 'anyone', 'anything', 'each', 'either', 'everybody', 'everyone', 'everything', 'neither', 'nobody', 'no one', 'nothing', 'one', 'somebody', 'someone', 'something', 'both', 'few', 'many', 'several', 'all', 'any', 'most', 'none', 'some', 'such']),
    'is_demonstrative_pronoun': lambda x: x.strip().lower() in set(['this', 'that', 'these', 'those']),
    'is_interrogative_pronoun': lambda x: x.strip().lower() in set(['who', 'whom', 'whose', 'which', 'what']),
    'is_relative_pronoun': lambda x: x.strip().lower() in set(['who', 'whom', 'whose', 'which', 'that']),
    'is_intensive_pronoun': lambda x: x.strip().lower() in set(['myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'yourselves', 'themselves']),
    'is_nonreflexive_pronoun': lambda x: x.strip().lower() in set(['i', 'me', 'we', 'us', 'you', 'he', 'him', 'she', 'her', 'it', 'they', 'them']),
}

STARTS_FEATURES = {
    'starts_w_vowel': lambda x: x.strip().lower().startswith(tuple(['a', 'e', 'i', 'o', 'u'])),
    'starts_w_consonant': lambda x: x.strip().lower().isalpha() and not x.strip().lower().startswith(tuple(['a', 'e', 'i', 'o', 'u'])),
    'start_w_no_space': lambda x: len(x) > 0 and x[0] != ' ',
    'start_w_no_space_and_letter': lambda x: len(x) > 0 and x[0].isalpha(),
    'start_w_no_space_and_digit': lambda x: len(x) > 0 and x[0].isdigit(),
    'starts_w_digit': lambda x: x.strip().isdecimal() and len(x.strip()) > 0,
    'starts_w_space': lambda x: x[0] == ' ' if len(x.strip()) > 0 else False,
    'start_w_space_and_letter': lambda x: len(x) > 1 and x[0] == ' ' and x[1].isalpha(),
    'starts_w_cap': lambda x: x.strip()[0].isupper() if len(x.strip()) > 0 else False,
    'starts_w_no_space_and_cap': lambda x: x[0].isupper() if len(x.strip()) > 0 else False,
    'starts_w_no_space_and_lower': lambda x: x[0].islower() if len(x.strip()) > 0 else False,
}


def suffix_features(x, suffix):
    return x.strip().lower().endswith(suffix)


# SUFFIX/PREFIX FEATURES
SUFFIXES = [
    "ness", "ment", "tion", "sion", "ance", "ence", "er", "or", "ist", "ism",
    "hood", "ship", "age", "able", "ible", "al", "ial", "ish", "ive", "less",
    "ful", "ous", "ic", "ly", "ward", "wise", "ize", "ise", "en", "ify", "let",
    "ling", "ie", "ette", "dom", "logy", "graphy", "phobia", "y", "cy", "ty",
    "ant", "ent", "ary", "ery", "ing", "ed", "s", "est"
]

SUFFIX_FEATURES = {
    f'end_w_{suffix}': partial(suffix_features, suffix=suffix)
    for suffix in SUFFIXES
}


def prefix_features(x, prefix):
    return x.strip().lower().startswith(prefix)


PREFIXES = ["un", "re", "in", "im", "dis", "en", "em", "non", "over", "mis", "sub", "pre", "inter", "fore", "de", "trans", "super", "semi", "anti", "mid", "under", "mono", "bi", "tri", "multi", "poly", "ex", "micro", "macro",
            "auto", "tele", "homo", "hetero", "post", "geo", "aero", "hydro", "thermo", "electro", "photo", "pan", "micro", "mega", "kilo", "pseudo", "neo", "anti", "counter", "ultra", "infra", "circum", "peri", "hemi", "ped", "pod", "bio",
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"
            ]

PREFIX_FEATURES = {
    f'start_w_{prefix}': partial(prefix_features, prefix=prefix)
    for prefix in PREFIXES
}

# is_month
MONTH_WORDS = set([
    'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
    'september', 'october', 'november', 'december',
    'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sept', 'oct', 'nov', 'dec',
])

# is_day of week
DAY_OF_WEEK_WORDS = set([
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'mon', 'tues', 'wed', 'thurs', 'fri', 'sat', 'sun',
])

# is_state
STATE_WORDS = set([
    'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado',
    'connecticut', 'delaware', 'florida', 'georgia', 'hawaii', 'idaho',
    'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana',
    'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota',
    'mississippi', 'missouri', 'montana', 'nebraska', 'nevada',
    'hampshire', 'jersey', 'mexico', 'york',
    'carolina', 'dakota', 'ohio', 'oklahoma', 'oregon',
    'pennsylvania', 'rhode',
    'tennessee', 'texas', 'utah', 'vermont', 'virginia', 'washington',
    'west virginia', 'wisconsin', 'wyoming',
])

STATE_ABV_WORDS = set([
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
])

WEBSITE_SUFFIXES = set([
    'com', 'net', 'org', 'edu', 'gov', 'mil', 'int', 'io', 'co', 'uk', 'ca', 'ai'
])

CONTRASTIVE_CONJUNCTIONS = set([
    'but', 'however', 'nonetheless', 'yet', 'nevertheless', 'although'
])

# is_country

# is_name

# measurement units
LENGTH_WORDS = set([
    'meter', 'metre', 'inch', 'foot', 'feet', 'yard', 'mile'
])

TIME_WORDS = set([
    'second', 'minute', 'hour', 'day', 'week', 'month', 'year'
])


WORD_GROUP_FEATURES = {
    'is_month': lambda x: x.strip().lower() in MONTH_WORDS,
    'is_day_of_week': lambda x: x.strip().lower() in DAY_OF_WEEK_WORDS,
    'is_state': lambda x: x.strip().lower() in STATE_WORDS,
    'is_state_abv': lambda x: x.strip() in STATE_ABV_WORDS,
    'is_website_suffix': lambda x: x in WEBSITE_SUFFIXES,
    'is_length': lambda x: any([word in x.strip().lower() for word in LENGTH_WORDS]),
    'is_time': lambda x: any([word in x.strip().lower() for word in TIME_WORDS]),
    'is_contrastive_conjuction': lambda x: x.strip().lower() in CONTRASTIVE_CONJUNCTIONS,
}


ALL_FEATURES = {
    **TYPE_FEATURES,
    **SYMBOL_FEATURES,
    **NUMERIC_FEATURES,
    **PRONOUN_FEATURES,
    **STARTS_FEATURES,
    **SUFFIX_FEATURES,
    **PREFIX_FEATURES,
    **WORD_GROUP_FEATURES,
}


def compute_token_dataset_statistics(vdf, token_tensor, smooth=True):
    vocab_size = len(vdf)
    token_count = torch.bincount(token_tensor.flatten(), minlength=vocab_size)
    if smooth:
        token_count += 1
    token_freq = token_count / token_count.sum()

    # TODO: document frequency statistics or other variance measures

    return token_freq


def make_vocab_df(model, small_norm_threshold=0.52):
    decoded_vocab = {
        tix: model.tokenizer.decode(tix)
        for tix in model.tokenizer.get_vocab().values()
    }

    vocab_df = pd.DataFrame({'token_string': decoded_vocab})
    for feature_name, feature_fn in ALL_FEATURES.items():
        vocab_df[feature_name] = vocab_df['token_string'].apply(feature_fn)

    vocab_df['unembed_norm'] = model.W_U.norm(
        dim=0).cpu().numpy()[:len(vocab_df)]
    vocab_df['embed_norm'] = model.W_E.norm(
        dim=1).cpu().numpy()[:len(vocab_df)]

    vocab_df['small_norm'] = vocab_df['embed_norm'] < small_norm_threshold

    empty_cols = vocab_df.sum(axis=0) == 0
    # drop columns that are all zero
    vocab_df = vocab_df.drop(columns=empty_cols[empty_cols].index)

    return vocab_df.copy()


def create_normalized_vocab(vocab_df, decoded_vocab):
    # create index of unique tokens when lowercased and stripped
    normed_vocab = {}
    for i in range(len(vocab_df)):
        norm_vocab = decoded_vocab[i].lower().strip()
        if norm_vocab not in normed_vocab:
            normed_vocab[norm_vocab] = [i]
        else:
            normed_vocab[norm_vocab].append(i)

    decoded_norm_vocab = {}
    token_ix_2_normed_ix = {}
    for normed_ix, (normed_str, token_ixs) in enumerate(normed_vocab.items()):
        for token_ix in token_ixs:
            token_ix_2_normed_ix[token_ix] = normed_ix
        decoded_norm_vocab[normed_ix] = normed_str

    return decoded_norm_vocab, token_ix_2_normed_ix


def get_unigram_df(dataset_df, decoded_norm_vocab, token_ix_2_normed_ix):
    dataset_df['normed_token'] = dataset_df['token'].apply(
        lambda x: token_ix_2_normed_ix[x])
    normed_vocab_count = dataset_df.normed_token.value_counts()
    normed_vocab_count_candidates = normed_vocab_count[
        normed_vocab_count > 3000].index.values

    normed_vocab_compression = pd.Series(
        list(token_ix_2_normed_ix.values())).value_counts()
    normed_vocab_unigram_candidates = normed_vocab_compression[
        normed_vocab_compression > 3].index.values

    normed_vocab_unigrams = set(normed_vocab_count_candidates) & set(
        normed_vocab_unigram_candidates)

    unigram_df = pd.DataFrame(
        np.zeros((len(token_ix_2_normed_ix), len(
            normed_vocab_unigrams)), dtype=bool),
        columns=[
            decoded_norm_vocab[i] + '_unigram'
            for i in normed_vocab_unigrams
        ]
    )
    for k, v in token_ix_2_normed_ix.items():
        if v in normed_vocab_unigrams:
            unigram_df.loc[k, decoded_norm_vocab[v] + '_unigram'] = True

    return unigram_df
