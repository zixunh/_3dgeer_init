#!/bin/bash
echo "CUDA_ARCH $1"

cp ./Dockerfile_$1 ./Dockerfile
docker -D build -t zixunh/3dgeer:latest .