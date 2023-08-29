
from argparse import ArgumentParser
from dataclasses import dataclass
from time import time
from langchain.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from src import logger

@dataclass
class Config:
    file_path: str
    start_page: int
    end_page: int   
    store: bool
    question: str


def clean_page(page: Document):
    content = page.page_content
    lines = content.split("\n")
    header = lines[0]
    if "Chapter" in header or "Item" in header:
        clean_content = "\n".join(lines[1:])
        page.page_content = clean_content
    return page

def get_knowledge_by_pdf():
    loader = PyPDFLoader(file_path=config.file_path)
    pages = loader.load()
    logger.info(f"Data loaded successuflly: {len(pages)} pages")
    filtered_pages = [clean_page(page) for page in pages if config.start_page <= page.metadata["page"] <= config.end_page]
    logger.info(f"Filtred page in the following rand: [{config.start_page}, {config.end_page}]")
    return filtered_pages

def get_knowledge_by_url():
    urls = []
    for i in range(10):
        urls.append(f"https://en.wikiquote.org/wiki/Pok%C3%A9mon/Season_{i}")
    loader = UnstructuredURLLoader(urls)
    pages = loader.load()
    logger.info(f"Data loaded successuflly: {len(pages)} pages")
    return pages

def set_embedding(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    chunks = text_splitter.split_documents(data)
    logger.info(f"Splitted documents into {len(chunks)} chunks")
    Chroma.from_documents(
        documents=chunks, 
        embedding=OpenAIEmbeddings(), 
        persist_directory="db"
    )
    logger.info("Chunks indexed into Chroma")

def get_embeddings():
    db = Chroma(persist_directory="db", embedding_function=OpenAIEmbeddings())
    return db.similarity_search(query=config.question, k=3)

def main(config: Config):
    if (config.store):
        pages = get_knowledge_by_url()
        set_embedding(pages)
        pdf = get_knowledge_by_pdf()
        set_embedding(pdf)
    else:
        learning_retriever = get_embeddings()
 
        llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0)
        logger.info("Generating answer with LLM")

        template_question = """Answer the question using the personality of ash character from the pokemon anime.
        Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. 
        {context}
        Question: {question}
        Answer in Portuguese with very didactically. if possible, use the character's adventures to help explain:"""

        QA_CHAIN_PROMPT = PromptTemplate(template=template_question, input_variables=["context", "question"])

        chain = load_qa_chain(llm, chain_type="stuff", prompt=QA_CHAIN_PROMPT)

        t1 = time()
        answer = chain({"input_documents": learning_retriever, "question": config.question }, return_only_outputs=True)
        t2 = time()
        print(f"elapsed time: {t2-t1}")

        print(answer["output_text"])

if __name__ == "__main__":
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--store", type=bool, default=False)
    argument_parser.add_argument("--question", type=str, default='')
    argument_parser.add_argument("--file_path", type=str, default='./historia-do-brasil.pdf')
    argument_parser.add_argument("--start_page", type=int, default=23)
    argument_parser.add_argument("--end_page", type=int, default=63)
    args = argument_parser.parse_args()
    config = Config(**vars(args))
    main(config)
