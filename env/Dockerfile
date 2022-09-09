FROM vincentcoriou/cube:latest
COPY . /opt/app
WORKDIR /opt/app
RUN pip install -r requirements.txt
RUN python -m spacy download en
