# Stereo Matching disparity estimation with custimized PsmNet

calcurate disparity with custmized PsmNet more light than ordinaly.
input right and left image and out put disparity 


<img src="https://user-images.githubusercontent.com/48679574/188292806-ec228e5f-c8f8-4320-b6a8-f42a789dde80.jpg" width="800px">

# model 


PsmNet more light than ordinaly. With SelfAttetion Laayer conv2d are far less than ordinally. and its accuracy is also goot.


<img src="https://user-images.githubusercontent.com/48679574/188292813-fa872d88-f893-472c-995d-1b790a87bcf2.png" width="800px">



| Model | CNN Arch | model param |
| :---         |     :---:      |        ---: |
| PSMNet| 77 2Dconv | 17,652,928 (5355840)|
| PSMNetplus | 29 2Dconv + SelfAttention| 12,658,784 (3775338)|




# Performance




# References
- [A Large-Scale Dataset for Stereo Matching in Autonomous Driving Scenarios](https://drivingstereo-dataset.github.io)
- [PSMNet](https://github.com/KinglittleQ/PSMNet)

