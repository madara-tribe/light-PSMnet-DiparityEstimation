# Stereo Matching disparity estimation with custimized PsmNet

calcurate disparity with custmized PsmNet more light than ordinaly.

input right and left image and output disparity 




<img src="https://user-images.githubusercontent.com/48679574/188292806-ec228e5f-c8f8-4320-b6a8-f42a789dde80.jpg" width="700px">


# model 


With SelfAttetion Layer, PsmNet become more light than ordinaly.

SelfAttetion Layer make conv2d layers far less than ordinally one and its accuracy is also goot.


<img src="https://user-images.githubusercontent.com/48679574/188292813-fa872d88-f893-472c-995d-1b790a87bcf2.png" width="700px">



| Model | CNN Arch | model param |
| :---         |     :---:      |        ---: |
| PSMNet| 77 2Dconv | 17,652,928 (5355840)|
| PSMNetplus(This model) | 29 2Dconv + SelfAttention| 12,658,784 (3775338)|




# Performance

<img src="https://user-images.githubusercontent.com/48679574/188294584-10c6d3cd-e345-4544-ae4d-b0b2438aa43c.jpeg" width="800px">

<img src="https://user-images.githubusercontent.com/48679574/188294587-6dd4f599-40ad-48c4-8e2d-558d7030b773.jpeg" width="800px">

<img src="https://user-images.githubusercontent.com/48679574/188294588-3814fb7f-2e01-48e3-9b08-740fb559f107.png" width="400px">

## pretrained model 

"StereoMatching in Autonomous Driving Scenarios" with 20 epoch [here](https://drive.google.com/file/d/1Ndhdwqs7eWPqptS4ww0uNh8_iLDmefq1/view?usp=sharing)

# References
- [A Large-Scale Dataset for Stereo Matching in Autonomous Driving Scenarios](https://drivingstereo-dataset.github.io)
- [Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318)
- [Pyramid Stereo Matching Network](https://arxiv.org/pdf/1803.08669.pdf)

