FROM tensorflow/tensorflow:2.5.1-gpu

WORKDIR /deployments

COPY . .

RUN apt update -y && sudo apt upgrade -y && \
    apt-get install -y wget build-essential checkinstall  libreadline-gplv2-dev  libncursesw5-dev  libssl-dev  libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev && \
    cd /usr/src && \
    sudo wget https://www.python.org/ftp/python/3.9.5/Python-3.9.5.tgz && \
    sudo tar xzf Python-3.9.5.tgz && \
    cd Python-3.9.5 && \
    sudo ./configure --enable-optimizations && \
    sudo make altinstall

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "run_transcripts_bert.py"]
