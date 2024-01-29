import argparse
import subprocess
import os
import sys
import requests
import logging

from gtts import gTTS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def pdf_to_mmd(input_file):
    logging.info(f"Extracting text from {input_file}.")
    command = ["nougat", input_file, "-o", "./"]
    try:
        subprocess.run(command, check=True)
    except:
        logging.exception(f"Error while extracting text from {input_file}.")
        sys.exit(1)


class MmdDocument:
    def __init__(self, mmd_file):
        self.file = mmd_file
        self.n_sec = 0
        self.document = {}
        self.has_title = False
        self.has_abstract = False
        self.has_sections = False
        self.has_misc = False
        self.parse_document()

    def parse_document(self):
        with open(self.file) as f:
            line = f.readline()
            while line:
                if self.is_title_start(line):
                    self.has_title = True
                    self.document["title"] = line
                    line = f.readline()
                    while not (
                        self.is_abstract_start(line)
                        or self.is_section_start(line)
                        or self.is_eod(line)
                    ):
                        self.document["title"] += line
                        line = f.readline()
                elif self.is_abstract_start(line):
                    self.has_abstract = True
                    self.document["abstract"] = line
                    line = f.readline()
                    while not (
                        self.is_title_start(line)
                        or self.is_section_start(line)
                        or self.is_eod(line)
                    ):
                        self.document["abstract"] += line
                        line = f.readline()
                elif self.is_section_start(line):
                    self.has_sections = True
                    self.document[f"section_{self.n_sec}"] = line
                    line = f.readline()
                    while not (
                        self.is_title_start(line)
                        or self.is_abstract_start(line)
                        or self.is_section_start(line)
                        or self.is_eod(line)
                    ):
                        self.document[f"section_{self.n_sec}"] += line
                        line = f.readline()
                    self.n_sec += 1
                else:
                    self.has_misc = True
                    self.document["misc"] = line
                    line = f.readline()
                    while not (
                        self.is_title_start(line)
                        or self.is_abstract_start(line)
                        or self.is_section_start(line)
                        or self.is_eod(line)
                    ):
                        self.document[f"misc"] += line
                        line = f.readline()

    @staticmethod
    def is_title_start(line):
        return line.startswith("# ")

    @staticmethod
    def is_abstract_start(line):
        return "###### Abstract" in line

    @staticmethod
    def is_section_start(line):
        return line.startswith("## ")

    @staticmethod
    def is_eod(line):
        return line == ""


def summarize_text(mmd_doc, api_key):
    logging.info("Summarizing text using NVIDIA-LLAMA-2 model.")
    invoke_url = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/7b3e3361-4266-41c8-b312-f5e33c81fc92"
    fetch_url_format = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    session = requests.Session()

    payload = {
        "messages": [
            {
                "content": f"Summarize the following text: {mmd_doc.document['section_0']}",
                "role": "user",
            }
        ],
        "temperature": 0.5,
        "top_p": 0.7,
        "max_tokens": 1024,
        "stream": False,
    }

    summary = ""
    for i in range(1, mmd_doc.n_sec):
        response = session.post(invoke_url, headers=headers, json=payload)

        while response.status_code == 202:
            request_id = response.headers.get("NVCF-REQID")
            fetch_url = fetch_url_format + request_id
            response = session.get(fetch_url, headers=headers)

        response.raise_for_status()
        response_body = response.json()["choices"][0]["message"]["content"]
        summary += " " + response_body
        payload["messages"][0][
            "content"
        ] = f"Summarize the following text: {mmd_doc.document[f'section_{i}']}"
    logging.info("Summary complete.")

    return summary


def create_audio(text, output_name):
    logging.info(f"Creating MP3 audio file '{output_name}'.")
    myobj = gTTS(text=text, lang="en", tld="com", slow=False)
    myobj.save(output_name)
    logging.info("Audio file created.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_file", help="Input PDF file to be summarized.")
    parser.add_argument("key", help="Personal key to use NVIDIA-LLAMA-2 API.")
    args = parser.parse_args()

    pdf_to_mmd(args.pdf_file)
    logging.info("Parsing extracted text from PDF file.")
    mmd_file = os.path.basename(args.pdf_file).replace(".pdf", ".mmd")
    mmd_doc = MmdDocument(mmd_file)
    summary = summarize_text(mmd_doc, api_key=args.key)

    audio_file = os.path.basename(args.pdf_file).replace(".pdf", ".mp3")
    create_audio(summary, audio_file)


if __name__ == "__main__":
    main()
