IMAGE=cnnlstmattn
run:
	docker build -t ${IMAGE} .
in:
	nvidia-docker run --runtime=nvidia -v /home/hagi:/home/app --rm -it ${IMAGE}:latest bash
