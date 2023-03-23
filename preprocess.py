import os
import argparse
import logging
import dill as pickle
import urllib
from tqdm import tqdm
import sys
import codecs
import spacy
import tarfile
import re
import torchtext.data
import random
import torchtext.datasets
from torchtext.datasets.translation import TranslationDataset
from learn_bpe import learn_bpe

__author__ = "Yu-Hsiang Huang"


class BPE(object):

    def __init__(self, codes, merges=-1, separator='@@', vocab=None, glossaries=None):

        codes.seek(0)
        offset=1

        # check version information
        firstline = codes.readline()
        if firstline.startswith('#version:'):
            self.version = tuple([int(x) for x in re.sub(r'(\.0+)*$','', firstline.split()[-1]).split(".")])
            offset += 1
        else:
            self.version = (0, 1)
            codes.seek(0)

        self.bpe_codes = [tuple(item.strip('\r\n ').split(' ')) for (n, item) in enumerate(codes) if (n < merges or merges == -1)]

        for i, item in enumerate(self.bpe_codes):
            if len(item) != 2:
                sys.stderr.write('Error: invalid line {0} in BPE codes file: {1}\n'.format(i+offset, ' '.join(item)))
                sys.stderr.write('The line should exist of exactly two subword units, separated by whitespace\n')
                sys.exit(1)

        # some hacking to deal with duplicates (only consider first instance)
        self.bpe_codes = dict([(code,i) for (i,code) in reversed(list(enumerate(self.bpe_codes)))])

        self.bpe_codes_reverse = dict([(pair[0] + pair[1], pair) for pair,i in self.bpe_codes.items()])

        self.separator = separator

        self.vocab = vocab

        self.glossaries = glossaries if glossaries else []

        self.glossaries_regex = re.compile('^({})$'.format('|'.join(glossaries))) if glossaries else None

        self.cache = {}

    def process_line(self, line, dropout=0):
        """segment line, dealing with leading and trailing whitespace"""

        out = ""

        leading_whitespace = len(line)-len(line.lstrip('\r\n '))
        if leading_whitespace:
            out += line[:leading_whitespace]

        out += self.segment(line, dropout)

        trailing_whitespace = len(line)-len(line.rstrip('\r\n '))
        if trailing_whitespace and trailing_whitespace != len(line):
            out += line[-trailing_whitespace:]

        return out

    def segment(self, sentence, dropout=0):
        """segment single sentence (whitespace-tokenized string) with BPE encoding"""
        segments = self.segment_tokens(sentence.strip('\r\n ').split(' '), dropout)
        return ' '.join(segments)

    def segment_tokens(self, tokens, dropout=0):
        """segment a sequence of tokens with BPE encoding"""
        output = []
        for word in tokens:
            # eliminate double spaces
            if not word:
                continue
            new_word = [out for segment in self._isolate_glossaries(word)
                        for out in encode(segment,
                                          self.bpe_codes,
                                          self.bpe_codes_reverse,
                                          self.vocab,
                                          self.separator,
                                          self.version,
                                          self.cache,
                                          self.glossaries_regex,
                                          dropout)]

            for item in new_word[:-1]:
                output.append(item + self.separator)
            output.append(new_word[-1])

        return output

    def _isolate_glossaries(self, word):
        word_segments = [word]
        for gloss in self.glossaries:
            word_segments = [out_segments for segment in word_segments
                                 for out_segments in isolate_glossary(segment, gloss)]
        return word_segments

def encode(orig, bpe_codes, bpe_codes_reverse, vocab, separator, version, cache, glossaries_regex=None, dropout=0):
    """Encode word based on list of BPE merge operations, which are applied consecutively
    """

    if not dropout and orig in cache:
        return cache[orig]

    if glossaries_regex and glossaries_regex.match(orig):
        cache[orig] = (orig,)
        return (orig,)

    if len(orig) == 1:
        return orig

    if version == (0, 1):
        word = list(orig) + ['</w>']
    elif version == (0, 2): # more consistent handling of word-final segments
        word = list(orig[:-1]) + [orig[-1] + '</w>']
    else:
        raise NotImplementedError

    while len(word) > 1:

        # get list of symbol pairs; optionally apply dropout
        pairs = [(bpe_codes[pair],i,pair) for (i,pair) in enumerate(zip(word, word[1:])) if (not dropout or random.random() > dropout) and pair in bpe_codes]

        if not pairs:
            break

        #get first merge operation in list of BPE codes
        bigram = min(pairs)[2]

        # find start position of all pairs that we want to merge
        positions = [i for (rank,i,pair) in pairs if pair == bigram]

        i = 0
        new_word = []
        bigram = ''.join(bigram)
        for j in positions:
            # merges are invalid if they start before current position. This can happen if there are overlapping pairs: (x x x -> xx x)
            if j < i:
                continue
            new_word.extend(word[i:j]) # all symbols before merged pair
            new_word.append(bigram) # merged pair
            i = j+2 # continue after merged pair
        new_word.extend(word[i:]) # add all symbols until end of word
        word = new_word

    # don't print end-of-word symbols
    if word[-1] == '</w>':
        word = word[:-1]
    elif word[-1].endswith('</w>'):
        word[-1] = word[-1][:-4]

    word = tuple(word)
    if vocab:
        word = check_vocab_and_split(word, bpe_codes_reverse, vocab, separator)

    cache[orig] = word
    return word

def recursive_split(segment, bpe_codes, vocab, separator, final=False):
    """Recursively split segment into smaller units (by reversing BPE merges)
    until all units are either in-vocabulary, or cannot be split futher."""

    try:
        if final:
            left, right = bpe_codes[segment + '</w>']
            right = right[:-4]
        else:
            left, right = bpe_codes[segment]
    except:
        #sys.stderr.write('cannot split {0} further.\n'.format(segment))
        yield segment
        return

    if left + separator in vocab:
        yield left
    else:
        for item in recursive_split(left, bpe_codes, vocab, separator, False):
            yield item

    if (final and right in vocab) or (not final and right + separator in vocab):
        yield right
    else:
        for item in recursive_split(right, bpe_codes, vocab, separator, final):
            yield item

def check_vocab_and_split(orig, bpe_codes, vocab, separator):
    """Check for each segment in word if it is in-vocabulary,
    and segment OOV segments into smaller units by reversing the BPE merge operations"""

    out = []

    for segment in orig[:-1]:
        if segment + separator in vocab:
            out.append(segment)
        else:
            #sys.stderr.write('OOV: {0}\n'.format(segment))
            for item in recursive_split(segment, bpe_codes, vocab, separator, False):
                out.append(item)

    segment = orig[-1]
    if segment in vocab:
        out.append(segment)
    else:
        #sys.stderr.write('OOV: {0}\n'.format(segment))
        for item in recursive_split(segment, bpe_codes, vocab, separator, True):
            out.append(item)

    return out


def read_vocabulary(vocab_file, threshold):
    """read vocabulary file produced by get_vocab.py, and filter according to frequency threshold.
    """

    vocabulary = set()

    for line in vocab_file:
        word, freq = line.strip('\r\n ').split(' ')
        freq = int(freq)
        if threshold == None or freq >= threshold:
            vocabulary.add(word)

    return vocabulary

def isolate_glossary(word, glossary):
    """
    Isolate a glossary present inside a word.
    Returns a list of subwords. In which all 'glossary' glossaries are isolated
    For example, if 'USA' is the glossary and '1934USABUSA' the word, the return value is:
        ['1934', 'USA', 'B', 'USA']
    """
    # regex equivalent of (if word == glossary or glossary not in word)
    if re.match('^'+glossary+'$', word) or not re.search(glossary, word):
        return [word]
    else:
        segments = re.split(r'({})'.format(glossary), word)
        segments, ending = segments[:-1], segments[-1]
        segments = list(filter(None, segments)) # Remove empty strings in regex group.
        return segments + [ending.strip('\r\n ')] if ending != '' else segments


_TRAIN_DATA_SOURCES = [
    {"url": "http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz",
     "trg": "news-commentary-v12.de-en.en",
     "src": "news-commentary-v12.de-en.de"},
]

_VAL_DATA_SOURCES = [
    {"url": "http://data.statmt.org/wmt17/translation-task/dev.tgz",
     "trg": "newstest2013.en",
     "src": "newstest2013.de"}]

_TEST_DATA_SOURCES = [
    {"url": "https://storage.googleapis.com/tf-perf-public/" \
            "official_transformer/test_data/newstest2014.tgz",
     "trg": "newstest2014.en",
     "src": "newstest2014.de"}]


# NLP Constants
PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def file_exist(dir_name, file_name):
    for sub_dir, _, files in os.walk(dir_name):
        if file_name in files:
            return os.path.join(sub_dir, file_name)
    return None


def download_and_extract(download_dir, url, src_filename, trg_filename):
    src_path = file_exist(download_dir, src_filename)
    trg_path = file_exist(download_dir, trg_filename)

    if src_path and trg_path:
        sys.stderr.write(f"Already downloaded and extracted {url}.\n")
        return src_path, trg_path

    compressed_file = _download_file(download_dir, url)

    sys.stderr.write(f"Extracting {compressed_file}.\n")
    with tarfile.open(compressed_file, "r:gz") as corpus_tar:
        corpus_tar.extractall(download_dir)

    src_path = file_exist(download_dir, src_filename)
    trg_path = file_exist(download_dir, trg_filename)

    if src_path and trg_path:
        return src_path, trg_path

    raise OSError(f"Download/extraction failed for url {url} to path {download_dir}")


def _download_file(download_dir, url):
    filename = url.split("/")[-1]
    if file_exist(download_dir, filename):
        sys.stderr.write(f"Already downloaded: {url} (at {filename}).\n")
    else:
        sys.stderr.write(f"Downloading from {url} to {filename}.\n")
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            urllib.request.urlretrieve(url, filename=filename, reporthook=t.update_to)
    return filename


def get_raw_files(raw_dir, sources):
    raw_files = {"src": [], "trg": [], }
    for d in sources:
        src_file, trg_file = download_and_extract(raw_dir, d["url"], d["src"], d["trg"])
        raw_files["src"].append(src_file)
        raw_files["trg"].append(trg_file)
    return raw_files


def mkdir_if_needed(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def compile_files(raw_dir, raw_files, prefix):
    src_fpath = os.path.join(raw_dir, f"raw-{prefix}.src")
    trg_fpath = os.path.join(raw_dir, f"raw-{prefix}.trg")

    if os.path.isfile(src_fpath) and os.path.isfile(trg_fpath):
        sys.stderr.write(f"Merged files found, skip the merging process.\n")
        return src_fpath, trg_fpath

    sys.stderr.write(f"Merge files into two files: {src_fpath} and {trg_fpath}.\n")

    with open(src_fpath, 'w') as src_outf, open(trg_fpath, 'w') as trg_outf:
        for src_inf, trg_inf in zip(raw_files['src'], raw_files['trg']):
            sys.stderr.write(f'  Input files: \n'
                             f'    - SRC: {src_inf}, and\n'
                             f'    - TRG: {trg_inf}.\n')
            with open(src_inf, newline='\n') as src_inf, open(trg_inf, newline='\n') as trg_inf:
                cntr = 0
                for i, line in enumerate(src_inf):
                    cntr += 1
                    src_outf.write(line.replace('\r', ' ').strip() + '\n')
                for j, line in enumerate(trg_inf):
                    cntr -= 1
                    trg_outf.write(line.replace('\r', ' ').strip() + '\n')
                assert cntr == 0, 'Number of lines in two files are inconsistent.'
    return src_fpath, trg_fpath


def encode_file(bpe, in_file, out_file):
    sys.stderr.write(f"Read raw content from {in_file} and \n" \
                     f"Write encoded content to {out_file}\n")

    with codecs.open(in_file, encoding='utf-8') as in_f:
        with codecs.open(out_file, 'w', encoding='utf-8') as out_f:
            for line in in_f:
                out_f.write(bpe.process_line(line))


def encode_files(bpe, src_in_file, trg_in_file, data_dir, prefix):
    src_out_file = os.path.join(data_dir, f"{prefix}.src")
    trg_out_file = os.path.join(data_dir, f"{prefix}.trg")

    if os.path.isfile(src_out_file) and os.path.isfile(trg_out_file):
        sys.stderr.write(f"Encoded files found, skip the encoding process ...\n")

    encode_file(bpe, src_in_file, src_out_file)
    encode_file(bpe, trg_in_file, trg_out_file)
    return src_out_file, trg_out_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-raw_dir', required=True)
    parser.add_argument('-data_dir', required=True)
    parser.add_argument('-codes', required=True)
    parser.add_argument('-save_data', required=True)
    parser.add_argument('-prefix', required=True)
    parser.add_argument('-max_len', type=int, default=100)
    parser.add_argument('--symbols', '-s', type=int, default=32000, help="Vocabulary size")
    parser.add_argument(
        '--min-frequency', type=int, default=6, metavar='FREQ',
        help='Stop if no symbol pair has frequency >= FREQ (default: %(default)s))')
    parser.add_argument('--dict-input', action="store_true",
                        help="If set, input file is interpreted as a dictionary where each line contains a word-count pair")
    parser.add_argument(
        '--separator', type=str, default='@@', metavar='STR',
        help="Separator between non-final subword units (default: '%(default)s'))")
    parser.add_argument('--total-symbols', '-t', action="store_true")
    opt = parser.parse_args()

    # Create folder if needed.
    mkdir_if_needed(opt.raw_dir)
    mkdir_if_needed(opt.data_dir)

    # Download and extract raw data.
    raw_train = get_raw_files(opt.raw_dir, _TRAIN_DATA_SOURCES)
    raw_val = get_raw_files(opt.raw_dir, _VAL_DATA_SOURCES)
    raw_test = get_raw_files(opt.raw_dir, _TEST_DATA_SOURCES)

    # Merge files into one.
    train_src, train_trg = compile_files(opt.raw_dir, raw_train, opt.prefix + '-train')
    val_src, val_trg = compile_files(opt.raw_dir, raw_val, opt.prefix + '-val')
    test_src, test_trg = compile_files(opt.raw_dir, raw_test, opt.prefix + '-test')

    # Build up the code from training files if not exist
    opt.codes = os.path.join(opt.data_dir, opt.codes)
    if not os.path.isfile(opt.codes):
        sys.stderr.write(f"Collect codes from training data and save to {opt.codes}.\n")
        learn_bpe(raw_train['src'] + raw_train['trg'], opt.codes, opt.symbols, opt.min_frequency, True)
    sys.stderr.write(f"BPE codes prepared.\n")

    sys.stderr.write(f"Build up the tokenizer.\n")
    with codecs.open(opt.codes, encoding='utf-8') as codes:
        bpe = BPE(codes, separator=opt.separator)

    sys.stderr.write(f"Encoding ...\n")
    encode_files(bpe, train_src, train_trg, opt.data_dir, opt.prefix + '-train')
    encode_files(bpe, val_src, val_trg, opt.data_dir, opt.prefix + '-val')
    encode_files(bpe, test_src, test_trg, opt.data_dir, opt.prefix + '-test')
    sys.stderr.write(f"Done.\n")

    field = torchtext.data.Field(
        tokenize=str.split,
        lower=True,
        pad_token=PAD_WORD,
        init_token=BOS_WORD,
        eos_token=EOS_WORD)

    fields = (field, field)

    MAX_LEN = opt.max_len

    def filter_examples_with_length(x):
        return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN

    enc_train_files_prefix = opt.prefix + '-train'
    train = TranslationDataset(
        fields=fields,
        path=os.path.join(opt.data_dir, enc_train_files_prefix),
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length)

    from itertools import chain
    field.build_vocab(chain(train.src, train.trg), min_freq=2)

    data = {'settings': opt, 'vocab': field, }
    opt.save_data = os.path.join(opt.data_dir, opt.save_data)

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    pickle.dump(data, open(opt.save_data, 'wb'))


def main_wo_bpe():
    '''
    Usage: python preprocess.py -lang_src de -lang_trg en -save_data multi30k_de_en.pkl -share_vocab
    '''

    spacy_support_langs = ['de', 'el', 'en', 'es', 'fr', 'it', 'lt', 'nb', 'nl', 'pt']

    parser = argparse.ArgumentParser()
    parser.add_argument('-lang_src', required=True, choices=spacy_support_langs)
    parser.add_argument('-lang_trg', required=True, choices=spacy_support_langs)
    parser.add_argument('-save_data', required=True)
    parser.add_argument('-data_src', type=str, default=None)
    parser.add_argument('-data_trg', type=str, default=None)

    parser.add_argument('-max_len', type=int, default=100)
    parser.add_argument('-min_word_count', type=int, default=3)
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-share_vocab', action='store_true')
    # parser.add_argument('-ratio', '--train_valid_test_ratio', type=int, nargs=3, metavar=(8,1,1))
    # parser.add_argument('-vocab', default=None)

    opt = parser.parse_args()
    assert not any([opt.data_src, opt.data_trg]), 'Custom data input is not support now.'
    assert not any([opt.data_src, opt.data_trg]) or all([opt.data_src, opt.data_trg])
    print(opt)

    src_lang_model = spacy.load(opt.lang_src)
    trg_lang_model = spacy.load(opt.lang_trg)

    def tokenize_src(text):
        return [tok.text for tok in src_lang_model.tokenizer(text)]

    def tokenize_trg(text):
        return [tok.text for tok in trg_lang_model.tokenizer(text)]

    SRC = torchtext.data.Field(
        tokenize=tokenize_src, lower=not opt.keep_case,
        pad_token=PAD_WORD, init_token=BOS_WORD, eos_token=EOS_WORD)

    TRG = torchtext.data.Field(
        tokenize=tokenize_trg, lower=not opt.keep_case,
        pad_token=PAD_WORD, init_token=BOS_WORD, eos_token=EOS_WORD)

    MAX_LEN = opt.max_len
    MIN_FREQ = opt.min_word_count

    if not all([opt.data_src, opt.data_trg]):
        assert {opt.lang_src, opt.lang_trg} == {'de', 'en'}
    else:
        # Pack custom txt file into example datasets
        raise NotImplementedError

    def filter_examples_with_length(x):
        return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN

    train, val, test = torchtext.datasets.Multi30k.splits(
        exts=('.' + opt.lang_src, '.' + opt.lang_trg),
        fields=(SRC, TRG),
        filter_pred=filter_examples_with_length)

    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    print('[Info] Get source language vocabulary size:', len(SRC.vocab))
    TRG.build_vocab(train.trg, min_freq=MIN_FREQ)
    print('[Info] Get target language vocabulary size:', len(TRG.vocab))

    if opt.share_vocab:
        print('[Info] Merging two vocabulary ...')
        for w, _ in SRC.vocab.stoi.items():
            # TODO: Also update the `freq`, although it is not likely to be used.
            if w not in TRG.vocab.stoi:
                TRG.vocab.stoi[w] = len(TRG.vocab.stoi)
        TRG.vocab.itos = [None] * len(TRG.vocab.stoi)
        for w, i in TRG.vocab.stoi.items():
            TRG.vocab.itos[i] = w
        SRC.vocab.stoi = TRG.vocab.stoi
        SRC.vocab.itos = TRG.vocab.itos
        print('[Info] Get merged vocabulary size:', len(TRG.vocab))

    data = {
        'settings': opt,
        'vocab': {'src': SRC, 'trg': TRG},
        'train': train.examples,
        'valid': val.examples,
        'test': test.examples}

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    pickle.dump(data, open(opt.save_data, 'wb'))


if __name__ == '__main__':
    main_wo_bpe()
