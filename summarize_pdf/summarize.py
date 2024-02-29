from dotenv import load_dotenv
import argparse as ap
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough

from gtts import gTTS

TEXT_CHUNK_SIZE = 10000


def concatenate_documents(documents: list) -> str:
    text = ""
    for document in documents:
        text += document.page_content
    return text


def extract_pdf_text(pdf_file: str) -> str:
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    return concatenate_documents(documents)


def split_text(text: str) -> list[str]:
    splitter = CharacterTextSplitter(
        separator=".\n",
        chunk_size=TEXT_CHUNK_SIZE,
        chunk_overlap=0,
        length_function=len,
    )
    text_chunks = splitter.split_text(text)

    # Sometimes CharacterTextSplitter returns text chunks larger than specified by chunk_size.
    # Here we exclude those so that there is no risk of inputs exceeding the
    # context window of our LLM.
    return [chunk for chunk in text_chunks if len(chunk) < TEXT_CHUNK_SIZE]


def get_summarization_chain():
    prompt_template = """You are an academic professor with many years of experience in summarizing academic texts with detailed description of the main points. Summarize the following text keeping a clear and in-depth description of the main ideas. Ignore parts of the text that contain useless information such as acknowledgements, list of publications, references, list of author names and institutions. If the text provided only contains useless information such as acknowledgements, or list of publications, or list of author names, or institutions return 'NOT USEFUL'. Provide the output without linebreaks.
    
    TEXT: {text}
    
    OUTPUT:"""
    prompt = ChatPromptTemplate.from_template(prompt_template)

    llm_summarizer = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.2)

    return {"text": RunnablePassthrough()} | LLMChain(llm=llm_summarizer, prompt=prompt)


def summarize_chunks(text_chunks: list[str]) -> str:
    summarization_chain = get_summarization_chain()
    summaries = summarization_chain.batch(text_chunks)
    summary = ""
    for piece in summaries:
        if piece["text"] != "NOT USEFUL":
            summary += " " + piece["text"]
    return summary


def create_audio_file(text: str, file_name: str):
    conclusion_text = "This summary was produced using generative artificial intelligence. Thank you for listening."
    final_summary = text + conclusion_text

    g_speech = gTTS(text=final_summary, lang="en", tld="com")
    g_speech.save(savefile=file_name)


def main():
    load_dotenv()

    parser = ap.ArgumentParser(description="Produce an audio summary of a PDF file.")
    parser.add_argument("pdf_file", help="Input PDF file to be summarized.")
    parser.add_argument(
        "-o",
        "--output",
        default="summary",
        help="Name of the output audio file without any audio extension. Defaults to 'summary'.",
    )
    args = parser.parse_args()

    if os.path.exists(args.pdf_file):
        text = extract_pdf_text(args.pdf_file)
        text_chunks = split_text(text)
        summary = summarize_chunks(text_chunks)
        audio_name = args.output + ".mp3"
        create_audio_file(summary, audio_name)


if __name__ == "__main__":
    main()
