# Audio summaries of academic papers

In the academic world, time is a valuable resource for researchers.
During my PhD studies, many times I would go through an entire scientific paper only to find out that the information it contained was not useful for my research.

This project tackles this problem by employing a large language model to summarize a paper and produce an MP3 file so that the main ideas of the paper can be conveniently listened to (e.g. during commute to work, a workout session or even doing the dishes).

The python script utilizes [NOUGAT (Neural Optical Understanding for Academic Documents)](https://github.com/facebookresearch/nougat/tree/main) to parse the input PDF file.
The extracted text is fed into the generative language model [NV-Llama2-70B-RLHF-Chat (by NVIDIA)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-foundation/models/nv-llama2-70b-rlhf) via its API, thereby producing a summary of the text.
In the final step, the summary is converted into an MP3 file employing [Google Translate’s Text-to-Speech API](https://gtts.readthedocs.io/en/latest/index.html).

An example PDF paper with the corresponding MP3 file containing the paper summary can be found in the `samples` directory.


## Prerequisites

Some packages must be installed for the script to work properly.
NOUGAT and Google's Text-to-Speech can be installed with
```
pip install nougat-ocr gTTS
```
In case of issues, please refer to the corresponding links provided above.

To utilize the API for the NV-Llama2-70B-RLHF-Chat model, it is necessary to generate an authorization key.
A key can be obtained in [this link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-foundation/models/nv-llama2-70b-rlhf/api) by clicking on the "Generate Key" button.
Remember to keep your personal key safe and never share it with anyone.


## Usage

Clone the repository to your local machine with
```
git clone git@github.com:allanpayeras/portifolio.git
```
Then change directory to the `summarize_pdf` directory.

To summarize a paper and generate the audio file run the command
```
python3 summarize.py path/to/pdf <authorization_key>
```
where `path/to/pdf` is the location of the PDF file and `<authorization_key>` should be replaced with the generated key for the NV-Llama2-70B-RLHF-Chat model.
The script will produce an MP3 file with the same name as the PDF input, but with the `.mp3` extension in the current working directory.