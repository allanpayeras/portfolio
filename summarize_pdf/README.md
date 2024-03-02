# Audio summaries of academic papers

In the academic environment, time is a particularly valuable resource for researchers.
During my doctoral studies, in some occasions, I would read through an entire scientific paper only to find out that the information it contained was not specially useful for my research.

This project tackles this problem by employing a large language model (OpenAI's GPT-3.5) to summarize scientific papers in PDF format and produce an MP3 audio file so that the main ideas of the paper can be conveniently listened to (e.g. during commute to work, a workout session or even while cleaning the house).
Furthermore, the developed tool can be useful for visually impaired individuals, aiding with inclusion in the scientific community.

Details of the implementation are given in the corresponding section below.

An example of a paper and its respective MP3 file containing the summary produced by the python script can be found in the `samples` directory.


## Prerequisites

Some packages must be installed for the python script to work properly.

The program heavily relies on the [LangChain](https://python.langchain.com/docs/get_started/introduction) framework, however some dependencies must be installed separately such as the `pypdf` and `langchain-openai`.
The `python-dotenv` package is necessary for loading the authentication key for the OpenAI API.
Lastly, we will need the [Google text-to-speech API](https://pypi.org/project/gTTS/).

Using `pip`, the command below should correctly install the dependencies for the script. 
```
pip install python-dotenv pypdf langchain langchain-openai gTTS
```
In some cases, `pip3` should be used instead of `pip`.
In case of trouble, please refer to the documentation of the corresponding packages.

Moreover, an authentication key for utilizing the OpenAI API is necessary.
One can be generated in the OpenAI API [website](https://openai.com/blog/openai-api).
After logging into your account (or signing up for an account), go to "API keys" (as shown in the image below) to generate a new key.

![open_ai_website](assets/openai_website.jpg)

Note that the utilization of the OpenAI API is not free. However, prices are very low. Summarizing academic papers only costs a few cents. More information about pricing can be found [here](https://openai.com/pricing).

For the script to work properly, a `.env` file should be created in the same directory.
Insert the generated key in the file as shown below
```
OPENAI_API_KEY=<your_key>
```
Remember to keep your personal key safe and never share it with anyone.


## Usage

Clone the repository to your local machine with
```
git clone git@github.com:allanpayeras/portfolio.git
```
Then change directory to the `summarize_pdf` directory.

To summarize a paper and generate the audio file run the command
```
python3 summarize.py path/to/pdf
```
where `path/to/pdf` is the location of the PDF file.
The script will produce an MP3 file with the name `summary.mp3` in the current working directory.
The default name of the audio file can be overridden with the option `-o`, however do not include the file extension in the provided name.


## Implementation details

To reach our goal of summarizing academic papers a sequence of steps were implemented in a python script.
Here a brief description is provided.
More information about the development process can be found in [this page](https://allanpayeras.github.io/portfolio/audio_summaries/audio_summaries/).

Initially, the provided PDF file is parsed using the [PyPDFLoader](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf) class, based on the PyPDF package, from the [LangChain framework](https://python.langchain.com/docs/get_started/introduction).
Subsequently, the entire text is divided into chunks of at maximum 12,000 characters each.

Utilizing LangChain, a chain was created for the summarization of individual text chunks.
The chain includes a prompt template that receives the text to be summarized and is subsequently fed to the large language model.
The model employed for this task is [GPT-3.5-turbo-0125](https://platform.openai.com/docs/models/gpt-3-5-turbo) from OpenAI where a temperature setting of 0.2 was applied.

With the created chain, LangChain conveniently allow us to summarize each text chunk (individually) in parallel.
The outputs of each individual summarization are merged to produce a final summary text.

In the final step, we utilized the [Google text-to-speech API](https://pypi.org/project/gTTS/) to effortlessly convert the summary text into an audio MP3 file.


## Improvements

Some aspects of this project can be improved to provide enhanced results.

* Improve the quality of the text extraction from the PDF file using a tool better suited for academic papers, such as [NOUGAT (Neural Optical Understanding for Academic Documents)](https://github.com/facebookresearch/nougat/tree/main).
The latter requires a GPU for practical performance, for this reason we chose the simpler `pypdf` package for now.

* Implement options for the user to control the length of the produced summary.
Most certainly it would require new summarization strategies including prompt engineering experimentation.

* The LLM could be fine-tuned for summarization of academic papers, yielding results more aligned with the needs of the academic community.

* The quality of the audio file can be enhanced using a more sophisticated text-to-speech (TTS) tool.
A test was performed utilizing the OpenAI TTS, yielding great [results](https://allanpayeras.github.io/portfolio/audio_summaries/audio_summaries/).
However, it substantially increases the cost.

* Provide a graphical user interface. A prominent option would be the implementation with [Streamlit](https://streamlit.io/).