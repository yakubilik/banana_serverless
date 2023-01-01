# Must use a Cuda version 11+
FROM yakupkeskin/sam_base:latest

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# We add the banana boilerplate here
ADD server.py ./src

ADD functions.py ./src

# Add your custom app code, init() and inference()
ADD app.py ./src

ADD Rotator.py ./src

ADD test2.py ./src

ADD asd2.png ./src

ADD data1.json ./src


EXPOSE 8000

WORKDIR ./src

CMD ["python3", "-u", "server.py"]

