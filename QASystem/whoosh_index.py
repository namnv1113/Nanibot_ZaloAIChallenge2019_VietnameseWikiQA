#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import whoosh.index as index
from whoosh.fields import ID, TEXT, STORED, Schema
from whoosh.analysis import LowercaseFilter, StopFilter
from argparse import ArgumentParser
from util import get_stopword_list, MyVietnameseTokenizer, get_encoding, get_scoring_algorithm, get_index_dir

doc_dir = ""
parser = ArgumentParser("Document indexing")
parser.add_argument("-m", dest="mode", help='Desired mode')
parser.add_argument("-f", dest="folder", help='The folder that stores the texts for indexing')

def get_document_names(dir):
    """ Get a list of all files in the folder and subfolders """
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dir)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dir, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_document_names(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

def get_schema():
    """ Return a schema used for indexing document """
    analyzer = MyVietnameseTokenizer() | LowercaseFilter() | StopFilter(get_stopword_list())
    return Schema(title=TEXT(analyzer=analyzer, stored=True, field_boost=1.5),
                  path=ID(unique=True, stored=True),
                  time=STORED,
                  content=TEXT(analyzer=analyzer, stored=True))

def add_doc(writer, filename):
    """ Add a document to the index """
    modify_time = os.path.getmtime(filename)
    title = os.path.basename(filename)
    title = title.split('.')[0]
    with open(filename, "r", encoding=get_encoding()) as file:
        content = file.read()
        writer.add_document(title=filename, path=filename, time=modify_time, content=content)

def clean_index(doc_dir):
    """ Create a brand new index (this will remove the old index) """
    # Create index from scratch (this will overwrite existing indexes)
    schema = get_schema()
    if not os.path.exists(get_index_dir()):
        os.mkdir(get_index_dir())
    ix = index.create_in(get_index_dir(), schema)   
    writer = ix.writer()
        
    for filename in get_document_names(doc_dir):
        add_doc(writer, filename)
    writer.commit(optimize=True)

def incremental_index(doc_dir):
    """ Update index based on document last update time """
    if (index.exists_in(get_index_dir()) == False):
        clean_index(doc_dir)
        return

    ix = index.open_dir(get_index_dir())

    indexed_paths = set()   # The set of all paths in the index
    to_index = set()        # The set of all paths we need to re-index
    writer = ix.writer()

    with ix.searcher() as searcher:
        # Loop over the stored fields in the index
        for fields in searcher.all_stored_fields():
            indexed_path = fields['path']
            indexed_paths.add(indexed_path)

            if not os.path.exists(indexed_path):
                # This file was deleted since it was indexed --> Delete index
                writer.delete_by_term('path', indexed_path)
            else:
                # Check if this file was changed since it was indexed
                indexed_time = fields['time']
                modify_time = os.path.getmtime(indexed_path)
                if modify_time > indexed_time:
                    # The file has changed, delete it and add it to the list of files to reindex
                    writer.delete_by_term('path', indexed_path)
                    to_index.add(indexed_path)

    for filename in get_document_names(doc_dir):
        path = os.path.join(doc_dir, filename)
        if path in to_index or path not in indexed_paths:
            add_doc(writer, filename)
    writer.commit(optimize=True)

if __name__ == "__main__":
    args = parser.parse_args()
    doc_dir = args.folder
    if (args.mode == 'create'):
        clean_index(doc_dir)
    elif (args.mode == 'update'):
        incremental_index(doc_dir)
    else:
        print("Unknown mode")
        exit(0)