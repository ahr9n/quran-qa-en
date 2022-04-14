from haystack import Finder
from haystack.preprocessor.utils import convert_files_to_dicts
from haystack.reader.farm import FARMReader
from haystack.document_store.memory import InMemoryDocumentStore
from haystack.retriever.sparse import TfidfRetriever
import streamlit as st
from os import path
import pandas as pd

st.markdown("<center> <h1> 📜 Quran Question Answering System Using English Translations </h1> </center>", True)


@st.cache
def read_corpus():
    document_store = InMemoryDocumentStore()
    doc_dir = "quran"
    dicts = convert_files_to_dicts(dir_path=doc_dir, split_paragraphs=True)
    document_store.write_documents(dicts)
    return document_store


@st.cache(allow_output_mutation=True)
def retriever():
    document_store = read_corpus()
    tfidf_retriever = TfidfRetriever(document_store=document_store)
    return tfidf_retriever


question = st.text_input('Ask your question here: ')

if st.button('Ask'):
    with st.spinner('Reading all English translations of the Holy Quran...'):
        retriever = retriever()

        # For doing QA at scale (i.e.many docs instead of single paragraph), you can load the model also in haystack
        if not (path.exists('data/mlm-temp')):
            reader = FARMReader(model_name_or_path="deepset/minilm-uncased-squad2", use_gpu=False)
            reader.save(directory='data/mlm-temp')
            st.info('Starting the query with a fresh model')
        else:
            reader = FARMReader(model_name_or_path="data/mlm-temp", use_gpu=False)
            st.info('Re-using the previous model')

        finder = Finder(reader, retriever)

        prediction = finder.get_answers(question=question, top_k_retriever=10, top_k_reader=5)

        keys = ['answer', 'context', 'meta', 'probability', 'score']
        print(list(map(prediction.get, ['query'])))
        print("\n")
        answer_frame = pd.DataFrame.from_records([list(map(i.get, keys)) for i in prediction['answers']])
        answer_frame.columns = ['Answer', 'Reference', 'Surah', 'Probability', 'Score']
        answer_frame['Surah'] = answer_frame['Surah']
        st.table(answer_frame.T)

        del reader, prediction, finder

        st.success('Done!')
