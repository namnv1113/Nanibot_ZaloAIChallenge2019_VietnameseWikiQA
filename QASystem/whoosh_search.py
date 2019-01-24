#!/usr/bin/python
# -*- coding: utf-8 -*-

import whoosh.index as index
import os
from util import preprocess_question, question_tokens_to_query, keyword_expansion, get_encoding, get_scoring_algorithm, get_index_dir, get_good_tokens

search_limit = 3

class SingleResult():
    """ A simple class represent a search result """
    def __init__(self, hit):
        self._score = hit.score
        self._path = hit["path"]
        self._content = hit["content"]

    def score(self):
        return self._score
    
    def path(self):
        return self._path

    def content(self):
        return self._content

class SimpleSearchResults():
    """ A simple data structure for search results """
    def __init__(self, search_results):
        self._scored_length = search_results.scored_length()
        self._length = len(search_results)
        self._has_matched_terms = search_results.has_matched_terms()
        self._hits = []
        if search_results.has_matched_terms():
            self._matched_terms = str(search_results.matched_terms())
            for hit in search_results:
                self._hits.append(SingleResult(hit))

    def scored_length(self):
        return self._scored_length

    def length(self):
        return self._length

    def hits(self):
        return self._hits

    def has_matched_terms(self):
        return self._has_matched_terms

    def matched_terms(self):
        return self._matched_terms

def search(query, search_limit=search_limit):
    """ Search the document index based on the query """
    if (index.exists_in(get_index_dir()) == False):
        print('No index for document created yet.')
        return 

    ix = index.open_dir(get_index_dir())

    with ix.searcher(weighting=get_scoring_algorithm()) as searcher:
        results = searcher.search(query, limit=search_limit, terms=True)
        return SimpleSearchResults(results)

def run(question, isQueryExpand):
    """ Search the document index that take a question list as input """
    # Process question
    tokens = preprocess_question(question)
        
    # Expand tokens for better resuls
    good_tokens = get_good_tokens(tokens)
    if (isQueryExpand):
        keywords = keyword_expansion(good_tokens)
    else:
        keywords = []
        for text, _, _ in good_tokens:
            keywords.append([text])
            
    # From question tokens to query
    query = question_tokens_to_query(keywords)

    # Search document index for results
    results = search(query)

    final_results = []
    if results.has_matched_terms():
        for hit in results.hits():
            final_results.append([hit.content(), hit.score()])
    return final_results

if __name__ == "__main__":
        #question = u"Việc đăng ký thay đổi người đại diện theo pháp luật của công ty \
        #              trách nhiệm hữu hạn, công ty cổ phần được pháp luật quy định như thế nào?"
        
        question = u"Cho em hỏi chuẩn đầu ra tiếng anh là bao nhiêu?"
        # questions.append(u"Yêu cầu của trường phải có ngoại ngữ gì khi tốt nghiệp?")
        # questions.append(u"Cho em hỏi chương trình tiên tiến là gì vậy?")
        # questions.append(u"4.9 có đủ qua môn không ạ?")
        # questions.append(u"Nhiêu điểm qua môn ạ?")
        # questions.append(u"Rớt bao nhiêu % số tín chỉ bị hạ bậc tốt nghiệp?")
        # questions.append(u"giảm loại tốt nghiệp?")
        # questions.append(u"Số tín chỉ tích lũy tối thiểu của khoa CNPM để ra trường theo chương trình đào tạo là bao nhiêu?")
        # questions.append(u"Số tín chỉ tích lũy tối thiểu để ra trường theo chương trình đào tạo là bao nhiêu?")
        # questions.append(u"Vắng bao nhiêu % số buổi cấm thi?")
        # questions.append(u"Sinh viên môn CNPM phải học bao nhiêu môn bắt buộc?")
        # questions.append(u"Sinh viên phải học bao nhiêu môn bắt buộc?")
        # questions.append(u"Mail của sinh viên trường sau khi tốt nghiệp có thể sử dụng không?")
        # questions.append(u"Sinh viên khoa khoa học máy tính phải học bao nhiêu tín chỉ các môn tự chọn?")
        # questions.append(u"Sinh viên phải học bao nhiêu tín chỉ các môn tự chọn?")
        # questions.append(u"Không học anh văn 3 - nộp bằng có được không (đối với K10)?")
        # questions.append(u"Số tín chỉ tối thiểu trong 1 học kì là bao nhiêu, học thiếu có bị cảnh cáo không?")
        # questions.append(u"Cảnh cáo bao nhiêu lần thì bị đuổi học?")
        # questions.append(u"Toeic bao nhiêu điểm đủ điều kiện làm khóa luận tốt nghiệp?")
        # questions.append(u"Đóng học phí trễ bao nhiêu ngày sẽ bị cảnh cáo")
        # questions.append(u"Để tham gia vào các nhóm nghiên cứu phải làm thế nào?")

        results = run(question)
        qa_pairs = []

        if results.has_matched_terms():
            for hit in results.hits():
                qa_pairs.append((question, hit.content()))
