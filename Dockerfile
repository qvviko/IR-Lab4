# Specify base image
FROM python:3
ENV PYTHONUNBUFFERED 1
# Copy particular file
COPY requirements.txt /tmp/
# Execute commands inside container, so that you can customize it
RUN pip3 install -r /tmp/requirements.txt
# Create and cd to this directory, set default command to run container
WORKDIR /app
# Copy files from project dir into container’s folder
COPY ./ /app
EXPOSE 8080
CMD python3 app.py