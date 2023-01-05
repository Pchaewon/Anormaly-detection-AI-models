# Anormaly-detection-AI-models
## 1. Introduction
This project is an AI model for anomaly detection tailored to the characteristics of data. There are a total of four anomaly detection AI models, which can be selected and used according to the characteristics and size of data.

## 2. Citation
```bash
@InProceedings{Wyatt_2022_CVPR,
    author    = {Wyatt, Julian and Leach, Adam and Schmon, Sebastian M. and Willcocks, Chris G.},
    title     = {AnoDDPM: Anomaly Detection With Denoising Diffusion Probabilistic Models Using Simplex Noise},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {650-656}
}
```
## 3. Models
The four anomaly detection AI models can be run through __main.py__.
### 3-1. CNN
CNN is the basic method used in anomaly detection tasks. It was simply implemented using a 3-layer convolution layer, max_pooling, and fc layer. <br>

learning_rate : 0.0001<br>
optimizer : Adam<br>
Loss_Function : CrossEntropy<br>

The execution method of this model is as follows.
```bash
python main.py --model CNN
```

### 3-2. AutoEncoder
For given data, abnormal samples can be detected through the process of dimension reduction and restoration using Principal Component Analysis (PCA). , Autoencoder is typically used as a basis for Neural Network. Autoencoder proceeds with encoding that compresses the input and decoding that restores it closer to the original, and through this, it has an operation similar to PCA that can learn the main components of data in that only important information of data can be learned compressively. you can see it does. 

Using Autoencoder, you can learn the characteristics of the normal region, which is the main component of the data, even without data labels. At this time, if a normal sample is put into the learned autoencoder, it is restored well, so there is almost no difference between input and output. On the other hand, if an abnormal sample is put in, the autoencoder restores it like a normal sample, so there is a difference in the process of finding the difference between input and output. Because it occurs prominently, abnormal samples can be detected.<ㅠㄱ>

learning_rate : 0.0001<br>
optimizer : Adam<br>
Loss_Function : MSEEntropy<br>

The execution method of this model is as follows.
```bash
python main.py --model AE
```

### 3-3. AnoGAN
> [Paper](https://link.springer.com/chapter/10.1007/978-3-319-59050-9_12)

AnoGAN combines GAN with an unsupervised method, which is a method of learning only normal data without abnormal data.

<img width="100%" src="https://user-images.githubusercontent.com/30998582/210761700-f9d37eb6-9aa3-45bd-b192-0c050c13a5c4.PNG"/>

#Training 
1. As shown in Fig below, only normal data is entered into the GAN model to train the distribution of normal data.
2. Enter normal/unseen data into a well-trained GAN model and compare anomaly scores to classify them.

Train Generator
Randomly extract the z vector from the latent space, input it to the Generator to create a fake image, and input this fake image to the Discriminator to learn to trick D into determining that it is a real image.

At this time, the weight of D is not udpated.

Discriminator Learning
For the generated image fake image and the actual image real image, it learns to judge well as fake/real respectively.

At this time, the weight of G is not updated.


#How to get Anomaly score
When a certain image (real image) is input, the method to classify whether it is normal or abnormal is as follows.

1. Generate a fake image by inputting the randomly generated latent vector z to the generator.
2. Update the latent vector z by repeating the backpropagation of the anomaly loss 500 times.
3. The generated image by passing the vector z obtained from step 2 through the generator is judged to be the image most similar to the input image (real image), and an anomaly score is obtained by comparing the generated image with the input image.
4. The formula below calculates the anomaly score,R(x) is the residual loss, and D(x) is the same as Discrimination Loss. In the paper, lambda is set to 0.1 and used.


### 3-4. AnoDDPM
For more details about AnoDDPM's technology, please refer to the AnoDDPM paper in the link.
> [Official Project](https://julianwyatt.co.uk/anoddpm) | [Paper](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/html/Wyatt_AnoDDPM_Anomaly_Detection_With_Denoising_Diffusion_Probabilistic_Models_Using_Simplex_CVPRW_2022_paper.html)

In this project, the files related to AnoDDPM are diffusion.py and UNet.py in the model folder and .py files in the core file.
```bash
├── model
│   ├── core
│       ├── evaluation.py
│       ├── helpers.py
│       ├── simplex.py
│       ├── solver.py
│       └── utils.py
│   ├── UNet.py
│   └── diffusion.py
``` 

AnoDDPM is a model for Anomaly Detection based on Diffusion Model. It uses Simplex Noise, not Gaussian Mixture Noise, to compensate for the limitations that occur in the Anomaly Detection of DDPM, which is an existing Diffusion Model. Diffusion model using simplex noise has the advantage of generating a normal image well even for the low frequency region of an abnormal image.

In this project, the principle of AnoDDPM was also quoted to create an AnoDDPM model based on the diffusion model. Compared to existing generative models, the diffusion model shows superior performance in the generative domain. Therefore, it provides good performance image generation based on normal image features. Based on these images, it is compared with abnormal images, and if a certain threshold is exceeded, it is determined as an outlier.

The execution method of this model is as follows.
```bash
python main.py --model AnoDDPM
```

## 4. Datasets

## 5. Results
