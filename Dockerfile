FROM tensorflow/tensorflow:2.9.1-gpu

WORKDIR /deployments

COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "run_transcripts_bert.py"]
