FROM tensorflow/tensorflow:2.9.1-gpu

WORKDIR /deployments

ENV LANG=C.UTF-8 \
    LANGUAGE=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONPATH="${PYTHONPATH}:."

COPY . .

RUN pip install -r requirements.txt

LABEL org.opencontainers.image.manifest="application/vnd.docker.distribution.manifest.v2+json"


ENTRYPOINT ["/bin/bash", "-c", "python3 predict_transcripts_bert.py $APP_PARAMS"]
