FROM tensorflow/tensorflow:2.5.1-gpu

WORKDIR /deployments

COPY . .

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    python3.9 python3-pip python3.9-dev

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "run_transcripts_bert.py"]
