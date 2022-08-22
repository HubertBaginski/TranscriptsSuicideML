FROM tensorflow/tensorflow:2.5.1-gpu

WORKDIR /deployments

COPY . .

RUN apt-get update \
 && apt-get install software-properties-common \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt-get update \
 && apt-get install python3.9

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "run_transcripts_bert.py"]
