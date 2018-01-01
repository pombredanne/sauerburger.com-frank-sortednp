FROM ubuntu:16.04

RUN apt-get update \
  && apt-get install -y pandoc \
  && rm -rf /var/lib/apt/lists/*
