FROM tensorflow/tensorflow:2.5.1-gpu

WORKDIR /deployments

COPY . .

RUN yum install gcc openssl-devel bzip2-devel libffi-devel gzip make -y
RUN yum install wget tar -y
WORKDIR /opt
RUN wget https://www.python.org/ftp/python/3.9.5/Python-3.9.5.tgz
RUN tar xzf Python-3.9.5.tgz
WORKDIR /opt/Python-3.9.5
RUN ./configure --enable-optimizations
RUN make altinstall
RUN rm -f /opt/Python-3.9.5.tgz

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "run_transcripts_bert.py"]
