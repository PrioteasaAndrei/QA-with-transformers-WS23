FROM python:3.9

# TODO review stucture
WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get install git
RUN apt-get install curl
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs
RUN git lfs install

# TODO cache all models
RUN cd src
RUN git clone https://huggingface.co/NeuML/pubmedbert-base-embeddings
RUN cd ..
# RUN python ./utils/download_models.py
 
# TODO port expose
# TODO build

CMD python -m streamlit run ./src/streamlit_app.py