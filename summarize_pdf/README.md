# Audio summaries of academic papers

In the academic world, time is a valuable resource for researchers. During my PhD studies, many times I would use my time to read scientific papers only to find out that the information it contained was not useful for my research.

This project tackles this problem by employing a large language model to summarize a paper and produce an MP3 file so that the main ideas of the paper can be conveniently listened to (specialy during commute to work, a workout session or even doing the dishes).

The python script utilizes [NOUGAT (Neural Optical Understanding for Academic Documents)](https://github.com/facebookresearch/nougat/tree/main) to parse the input PDF file. The extracted text is fed into the generative language model [NV-Llama2-70B-RLHF-Chat (by NVIDIA)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-foundation/models/nv-llama2-70b-rlhf) via its API, thereby producing a summary of the text. In the final step, the summary is converted into a speech MP3 file employing [Google Translate’s Text-to-Speech API](https://gtts.readthedocs.io/en/latest/index.html).

## Prerequisites

## Usage