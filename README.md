# Stereo Matching disparity estimation with custimized PsmNet

calcurate disparity with custmized PsmNet more light than ordinaly.
input right and left image and out put disparity 


<img src="https://user-images.githubusercontent.com/48679574/188292724-648deeff-8c8f-46a7-8d1b-70c60f1e158c.jpeg" width="400px">

# model 


PsmNet more light than ordinaly. With SelfAttetion Laayer conv2d are far less than ordinally. and its accuracy is also goot.


<img src="https://user-images.githubusercontent.com/48679574/188292692-6a3f164c-6691-4847-b98e-f3fe38d75d89.png" width="400px">



| Model | CNN Arch | model param |
| :---         |     :---:      |        ---: |
| PSMNet| 77 2Dconv | 17,652,928 (5355840)|
| PSMNetplus | 29 2Dconv + SelfAttention| 12,658,784 (3775338)|




# Performance




# References
- [A Large-Scale Dataset for Stereo Matching in Autonomous Driving Scenarios](https://drivingstereo-dataset.github.io)
- [PSMNet](https://github.com/KinglittleQ/PSMNet)
