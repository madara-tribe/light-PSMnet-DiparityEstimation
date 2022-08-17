IMAGE=cnnlstmattn
run:
	docker build -t ${IMAGE} .
in:
	NV_GPU=4,5 nvidia-docker run --runtime=nvidia -v /home/hagi:/home/app --rm -it ${IMAGE}:latest bash
