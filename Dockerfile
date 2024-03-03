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
RUN git clone https://huggingface.co/NeuML/pubmedbert-base-embeddings ./src/pubmedbert-base-embeddings

RUN wget https://github.com/git-ecosystem/git-credential-manager/releases/download/v2.3.0/gcm-linux_amd64.2.3.0.deb
RUN dpkg -i gcm-linux_amd64.2.3.0.deb
RUN git-credential-manager configure
RUN git config --global --unset-all credential.helper
RUN git config --global credential.helper store
RUN --mount=type=secret,id=mysecret \
  HF_TOKEN=$(cat src/.env) \
  && export HF_TOKEN \
  && sh ./git_configure.sh
RUN git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf ./src/Llama-2-7b-chat-hf

# RUN python ./utils/download_models.py
 
# TODO port expose
# TODO build

CMD python -m streamlit run ./src/streamlit_app.py