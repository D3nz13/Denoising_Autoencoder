# Denoising Autoencoder
## Table of conents
* [Description](#description)
* [Used libraries](#used-libraries)
* [Model architecture](#model-architecture)
* [Graphs](#graphs)

## Description
This project has been made to create my very first autoencoder and to practice.  
The used dataset is a classic ```MNIST digit dataset```, available in built-in datasets in keras.  
The dataset contains 28x28 images of digits. I added random noise to each image and tried to reconstruct it by using an autoencoder.  
The encoded images are 7x7 images.  

## Used libraries
```tensorflow==2.5.0```  
```numpy==1.19.5```  
```matplotlib==3.4.1```  

## Model architecture
- Encoder  
![Encoder architecture](/graphs/encoder_summary.png)  
- Decoder  
![Decoder architecture](/graphs/decoder_summary.png)  

## Graphs
- Clean vs noisy data  
![Clean vs noisy data](/graphs/clean_vs_noisy_train.png)  
- Loss plot  
![Loss plot](/graphs/loss_plot.png)  
- Final results   
![Final results](/graphs/results_0.4.png)  
The ```noise factor``` was equal to ```0.4```  
The first row is the clean test data  
The second row is the noisy test data  
The third row is the encoded noisy test data  
The last row is the decoded noisy test data  
- Results for ```noise factor = 0.6```  
![Final results for noise factor = 0.6](/graphs/results_0.6.png)  
- Results for ```noise factor = 0.8```  
![Final results for noise factor = 0.8](/graphs/results_0.8.png)  
- Results for ```noise factor = 1.0```  
![Final results for noise factor = 1.0](/graphs/results_1.0.png)  