
FROM python:3.7-slim

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 8501

RUN mkdir ~/.streamlit

WORKDIR /app

ENTRYPOINT ["streamlit", "run"]

CMD ["read_streamlit.py"]