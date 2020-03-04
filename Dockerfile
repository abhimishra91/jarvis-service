FROM python:3.6

WORKDIR /usr/app

RUN apt-get update && \
    apt-get install --no-install-recommends -y vim 

RUN pip install Flask 
RUN pip install regex==2018.11.22
RUN pip install numpy scipy pandas
RUN pip install torch
RUN pip install transformers


COPY . /usr/app/
RUN ls -la /usr/app/

EXPOSE 80

CMD [ "python", "server.py" ]
