from python:3.8

RUN mkdir /app
WORKDIR /app
ADD . /app

RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --upgrade pip

RUN pip install -r /app/requirements.txt

RUN mkdir model
RUN mkdir data

#yolo model - all-exp16.pt
# https://drive.google.com/file/d/1fjCSnd7z1E6WK6ccQdA9LXcAD8Bl3jcT/view?usp=sharing
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1fjCSnd7z1E6WK6ccQdA9LXcAD8Bl3jcT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1fjCSnd7z1E6WK6ccQdA9LXcAD8Bl3jcT" -O model/all_exp16.pt && rm -rf /tmp/cookies.txt

#recognition model - client_samyak.pkl
# https://drive.google.com/file/d/10Q7tTLMF9lH9h3zNM99sqiyV-ToJj3Rj/view?usp=share_link
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=10Q7tTLMF9lH9h3zNM99sqiyV-ToJj3Rj' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=10Q7tTLMF9lH9h3zNM99sqiyV-ToJj3Rj" -O model/client_samyak.pkl && rm -rf /tmp/cookies.txt

#sorted dataset
# https://drive.google.com/drive/folders/1ZL1GQDG27cxsSvKnYRlYMF4Z3D8ElFCe?usp=sharing
# https://drive.google.com/file/d/1WcTcBR6o4A3LHSuylKYforM4DPR2OADq/view?usp=share_link
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1WcTcBR6o4A3LHSuylKYforM4DPR2OADq' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1WcTcBR6o4A3LHSuylKYforM4DPR2OADq" -O client_samyak_sorted.zip && rm -rf /tmp/cookies.txt

#unzip and save database into data folder
RUN unzip client_samyak_sorted.zip -d data

CMD ["python", "app.py"]