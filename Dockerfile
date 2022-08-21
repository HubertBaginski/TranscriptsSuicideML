FROM python:3.9-bullseye

ENV FOO="bar"

WORKDIR /deployments

COPY docker .

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "run_transcripts_bert.py"]