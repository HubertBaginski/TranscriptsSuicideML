FROM tensorflow/tensorflow:2.9.1-gpu

WORKDIR /deployments

ENV LANG=C.UTF-8 \
    LANGUAGE=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONPATH="${PYTHONPATH}:." \
    APP_DEFAULT_PARAMS="--checkpoint_folder /checkpoint_folder --epochs 2 --batch_size 4 --variable_code MF02_01"

COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "run_transcripts_bert.py $APP_DEFAULT_PARAMS $APP_PARAMS"]
