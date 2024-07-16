FROM python:3.10

RUN git clone https://github.com/huggingface/optimum-tpu.git

WORKDIR optimum-tpu

RUN pip install -e . -f https://storage.googleapis.com/libtpu-releases/index.html

RUN pip install trl peft

RUN pip install -U jupyterlab notebook

RUN pip install ipywidgets widgetsnbextension
