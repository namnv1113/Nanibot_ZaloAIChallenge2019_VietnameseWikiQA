#!/usr/bin/python
# -*- coding: utf-8 -*-

from pyvi import ViTokenizer, ViPosTagger
from whoosh.analysis import Token, Tokenizer
from whoosh.compat import text_type
import whoosh.index as index
from whoosh import scoring
import whoosh.qparser as qparser
from gensim.models import KeyedVectors
import json
import string
import os
from math import log

index_dir = "./index"
stopword_filepath = "./vietnamese-stopwords-dash.txt"
encoding = 'utf-8'
keywords_file = "./keywords.txt"
vi_wordvectors_file = "./wiki.vi.model.bin.gz"
score_algorithm = scoring.TF_IDF()
fuzzy_prefix_length = 1
fuzzy_postfix_length = 2
keyword_expansion_limit = 5
appear_percentage_upperlimit = 50
appear_percentage_lowerlimit = 0

# Functions to get parameters ##################################################################
def get_index_dir():
    return index_dir

def get_encoding():
    """ Base encoding use across all documents and searches """
    return encoding

def get_scoring_algorithm():
    """ Base scoring algorithm for index and search """
    return scoring.BM25F()

def my_tokenize_func(value):
    """ A rewritten tokenize function. Change this if you wish to change the tokenizer """
    return ViTokenizer.tokenize(value).split()

def my_postagging_func(value):
    """ A rewritten pos tagging function. Change this if you wish to change the tokenizer """
    return ViPosTagger.postagging(ViTokenizer.tokenize(value))

################################################################################################

# Classes & Functions rewritten ################################################################
class MyVietnameseTokenizer(Tokenizer):
    """ An adapted tokenizer to use in whoosh library (based on whoosh implemented class)"""
    def __call__(self, value, positions=False, chars=False, keeporiginal=False,
                 removestops=True, start_pos=0, start_char=0, tokenize=True,
                 mode='', **kwargs):
        """
        Rewritten call method
        :param value: The unicode string to tokenize.
        :param positions: Whether to record token positions in the token.
        :param chars: Whether to record character offsets in the token.
        :param start_pos: The position number of the first token.
        :param start_char: The offset of the first character of the first token. 
        :param tokenize: if True, the text should be tokenized.
        """

        assert isinstance(value, text_type), "%r is not unicode" % value

        t = Token(positions, chars, removestops=removestops, mode=mode,
                  **kwargs)
        if not tokenize:
            t.original = t.text = value
            t.boost = 1.0
            if positions:
                t.pos = start_pos
            if chars:
                t.startchar = start_char
                t.endchar = start_char + len(value)
            yield t
        else:
            # The default: expression matches are used as tokens
            for pos, match in enumerate(my_tokenize_func(value)):
                t.text = match
                t.boost = 1.0
                if keeporiginal:
                    t.original = t.text
                t.stopped = False
                if positions:
                    t.pos = start_pos + pos
                if chars:
                    t.startchar = start_char
                    t.endchar = start_char + len(match)
                    start_char = t.endchar + 1
                yield t
################################################################################################

# Get required datas ###########################################################################
def get_stopword_list(filename=stopword_filepath):
    """ Get a list of stopword from a file """
    with open(filename, 'r', encoding=encoding) as f:
        stoplist = [line for line in f.read().splitlines()]
    return stoplist
################################################################################################

# Text to query ################################################################################
def normalize_text(s):
    """ Remove unnecessary whitespace, puctation and change to lower """
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_punc(lower(s)))

def tokens_format(tokens, stoplist=None, minsize=2, maxsize=None, whitespace_subtitute=None):
    """ Remove stop words, words too short or too long and change space character (to _ for Vietnamese text) """
    if stoplist is None:
        stoplist = get_stopword_list()
    new_tokens = []
    token_length = len(tokens[0])
    for i in range(token_length):
        text = tokens[0][i]
        tag = tokens[1][i]
        if (whitespace_subtitute is not None):
            text = text.replace(" ",  whitespace_subtitute)
        if (len(text) >= minsize and (maxsize is None or len(text) <= maxsize) and text not in stoplist):
            new_tokens.append((text, tag))

    return new_tokens

def assign_appear_percentage(tokens):
    """ Assign an additional value indicate appearance count of each token """
    tokens_with_idf = []
    ix = index.open_dir(index_dir)
    with ix.searcher(weighting=score_algorithm) as searcher:
        dc = searcher.doc_count_all()
        for (text, pos) in tokens:
            # idf = searcher.idf("content", text)
            n = searcher.doc_frequency("content", text)
            tokens_with_idf.append((text, pos, float(n * 100 / dc)))
    return tokens_with_idf
    
def preprocess_question(question):
    """ Transform questions to tokens with pos tagging and idf count """
    _query = normalize_text(question)   
    token_with_tags = my_postagging_func(_query)
    tokens_filter = tokens_format(token_with_tags, whitespace_subtitute="_")  
    tokens_with_idf = assign_appear_percentage(tokens_filter)

    return tokens_with_idf

def get_good_tokens(tokens):
    """ A native function to define good tokens for searching 
        Currently get all Noun and Verb """
    result = []
    for (text, pos, idf) in tokens:
        if ('N' in pos or pos == 'V'):
            result.append((text, pos, idf))
    return result


def keyword_expansion(tokens, word2vec_file = vi_wordvectors_file, word2vec_is_binary = True, expansion_length=keyword_expansion_limit, idf_upperlimit = appear_percentage_upperlimit, idf_lowerlimit = appear_percentage_lowerlimit):
    """ Remove word with high appearance (since it rarely affect search result) and find synonym of each tokens with very low appearnce """
    vectors = KeyedVectors.load_word2vec_format(word2vec_file, binary=word2vec_is_binary)
    keywords = []
    for text, _, idf in tokens:
        keyword = []
        if (idf >= appear_percentage_upperlimit):       # Unimportant word appear too many times --> Remove
            continue
        elif (idf > appear_percentage_lowerlimit):      # Lowerlimit < idf < upperlimit
            keyword.append(text)
        else:
            # Find synonyms based on word2vec
            try:
                synonyms = vectors.most_similar(text, topn=keyword_expansion_limit)
                synonyms = assign_appear_percentage(synonyms)
                
                for (synonym, _, idf) in synonyms:
                    if (idf >= appear_percentage_upperlimit):      
                        continue
                    elif (idf > appear_percentage_lowerlimit):    
                        keyword.append(synonym)
            except KeyError:        # Word doesn't exist in word2vec
                pass

        if len(keyword) > 0:
            keywords.append(keyword)
    return keywords

def question_tokens_to_query(keywords):
    """ From a list of keywords and its synonym, transform to whoosh-defined query format """
    # Build query from keywords 
    query_str = ""
    for keyword in keywords:
        keywords_str = "("
        for i in range(len(keyword)):
            keywords_str += keyword[i] + " OR "
        keywords_str = keywords_str[:-4]    # Remove the last " OR "
        keywords_str += ")"
        query_str += keywords_str + " "

    # From query string build whoosh-defined query
    ix = index.open_dir(index_dir)
    parser = qparser.MultifieldParser(["title", "content"], ix.schema)
    parser.remove_plugin_class(qparser.PhrasePlugin)
    parser.add_plugin(qparser.SequencePlugin())     # For complex pharse query
    parser.add_plugin(qparser.FuzzyTermPlugin())    # Search for term that dont have to match exactly
    query = parser.parse(query_str)

    return query

################################################################################################

# # Sample run ###################################################################################
# if __name__ == "__main__":
#     # Testing purpose
#     tokenizer = MyVietnameseTokenizer()
#     with open("res_token.txt", "w+", encoding="utf-8") as f:
#         for token in tokenizer(u"Điểm trung bình học kỳ (ĐTBHK) là điểm trung bình có trọng số của các môn học mà sinh viên đăng ký học và được Trường xếp lớp trong học kỳ đó, với trọng số là số tín chỉ của mỗi môn học tương ứng. ĐTBHK được dùng để xét học bổng, khen thưởng, xử lý học vụ sau mỗi học kỳ. Điểm trung bình tích lũy toàn khóa (ĐTBTLTK) là điểm tính theo kết quả của các học phần đạt từ điểm 5,0 trở lên mà sinh viên đã đăng ký học tại Trường (kể cả các học phần có điểm bảo lưu). ĐTBTLTK được tính khi sinh viên đủ điều kiện tốt nghiệp, được dùng để phân loại kết quả học tập và xếp hạng tốt nghiệp. Điểm trung bình chung tích lũy (ĐTBCTL) là điểm trung bình của các môn học mà sinh viên đã đăng ký học từ lúc bắt đầu khóa học đến thời điểm được tính với trọng số là số tín chỉ của mỗi môn học (lấy điểm cao nhất trong các lần học của mỗi học phần). ĐTBCTL được dùng để xét số tín chỉ được phép đăng ký trong học kỳ, học vượt, xét chuyển ngành/chương trình, xét điều kiện làm khóa luận tốt nghiệp (dành cho hệ chính quy đại trà). Kết quả học tập của học kỳ hè (nếu có) được tính chung vào học kỳ liền kề trước đó. Học phần có kết quả từ 5,0 điểm trở lên được bảo lưu khi sinh viên học thêm một ngành học mới trong Trường. Điểm bảo lưu được tính vào ĐTBTLTK, ĐTBCTL của ngành học đó. Không tính kết quả thi các học phần Giáo dục quốc phòng-An ninh và Giáo dục thể chất vào ĐTBHK, ĐTBTLTK hoặc ĐTBCTL. Việc đánh giá kết quả và điều kiện cấp chứng chỉ đối với học phần này theo quy định riêng của Bộ GD&ĐT. Điểm I, điểm M không được tính trong ĐTBHK, ĐTBTLTK và ĐTBCTL. Ngoài ra điểm bảo lưu không được tính trong ĐTBHK.", positions=True, chars=True):
#             f.write(repr(token.text) + "\t" + str(token.pos) + " (" + str(token.startchar) + ", " + str(token.endchar) + ")\n")
# ################################################################################################
