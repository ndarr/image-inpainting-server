FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime
COPY ./requirements.txt /var/www/requirements.txt
RUN pip install --upgrade pip
RUN pip install cython
RUN pip install -r /var/www/requirements.txt
COPY *.py /app/
COPY model_hybrid_3x3_1.16loss.pt /app/
WORKDIR /app
CMD ["flask", "run", "-h", "0.0.0.0", "-p", "5000"]