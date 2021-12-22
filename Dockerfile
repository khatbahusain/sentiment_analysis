FROM python:3.7.6

WORKDIR /app


COPY ./src /app/src
COPY ./requirements.txt /app

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host=0.0.0.0", "--reload"]