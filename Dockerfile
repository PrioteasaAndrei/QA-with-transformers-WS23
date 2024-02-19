FROM python:3.10.12

# TODO review stucture
WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
 
# TODO port expose
# TODO build

CMD ["python", "./src/app.py"]