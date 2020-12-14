FROM continuumio/anaconda3

WORKDIR /home/PM2.5_Web_App

COPY geostreamlit.yml ./
COPY app.py ./
COPY boot.sh ./
COPY all_v4_2 ./
COPY images ./
COPY latlon_dict.csv ./
COPY LICENSE ./
COPY Procfile ./
COPY README.md ./
COPY setup.sh ./

RUN chmod +x boot.sh

RUN conda env create -f geostreamlit.yml

RUN echo "source activate your-environment-name" &gt; ~/.bashrc
ENV PATH /opt/conda/envs/geostreamlit/bin:$PATH

EXPOSE 5000

ENTRYPOINT ["./boot.sh"]