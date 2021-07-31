# CIFAR10 GAN

A Deep Convolutional Generative Adversarial Network (DCGAN) was used to generate synthetic images from each class of the CIFAR10 dataset.

![generator](https://user-images.githubusercontent.com/60960803/127735392-32fc4bc8-bab8-4a73-b6f4-96c4c3dce49d.png)

Generator architecture
#
![discriminator](https://user-images.githubusercontent.com/60960803/127735394-4b258f0e-7990-4fae-9978-b8d83e098068.png)

Discriminator architecture
#
![gan](https://user-images.githubusercontent.com/60960803/127735395-cd39e5b9-e3ed-485c-873e-92cadcd4fb5b.png)

Whole GAN architecture
#
LeakyReLU was used as the activation function of each layer. tanh was used as the activation of the output layer of the generator. Feel free to check `GAN.py` for more information.
#
![images](https://user-images.githubusercontent.com/60960803/127735191-e8ea1ee8-e49e-4d3a-93a8-0a3971693026.png)

Here are sample images of each class. `generate_data.py` was used to generate these images, as well as the new data in csv form (saved in `dataset/new_data/`. 
However, those csv files weren't included in this repo because the total file size is 5GB, but it should be generated in a few seconds after training the GAN in `generate_data.py`. 

A new instance of the GAN was used to train the classes separately to avoid mode collapse. Also, the data was already sorted into their respective classes this way. 
However, it looks like the GAN still suffered from mode collapse when training for the `dog` class (we can see that blue is a dominant color in each image). 
Lastly, the overall quality of the generated images is quite bad as training a CNN which gave 81% accuracy using the training set only gave 21% accuracy when trained using the generated images 
(the code of this training isn't included in this repo, but the training process is uploaded [here](https://github.com/kevinesg/kaggle_competitions/tree/main/CIFAR10)).

The area of improvement is the model structure of the GAN, most likely the generator more than the discriminator. 
No further fine-tuning was attempted as the main purpose of this project is familiarity with GANs using an RGB-image dataset.
#
If you have any questions or suggestions, feel free to contact me here. Thanks for reading!
