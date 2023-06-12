# senior-design-project_23
## Title:
ENHANCING SPEECH IN REAL-TIME: USING DENSE NEURAL NETWORKS WITH DILATED CONVOLUTIONS AND SQUEEZED TEMPORAL CONVOLUTIONAL MODULES IN THE TIME DOMAIN.
### Abstract:
we present a fully convolutional neural network for enhancing real-time speech in the time domain. Skip connections are included in the architecture of the proposed encoder-decoder network. The layers in the decoder and encoder are down the lined by densely connected blocks (DCB) made of causal and dilated convolutions. These dilated convolutions facilitate the aggregation of contextual information across multiple resolutions. The network is ideal for real-time applications due to the causal convolutions' utilization of information inflow prevention from subsequent frames. Additionally, we propose employing up sampling in the decoder with sub-pixel convolutional layers. We also proposed employing a Squeezed temporal convolutional network (STCNs) after every dense block in encoder and decoder. To train the model, a loss function consisting of two components is also utilized: a loss in both the frequency domain and the time domain. The model performs better than the time-domain loss using the proposed loss function. According to experimental outcomes, the suggested model greatly surpasses previous state-of-the-art models in quality scores as well as objective intelligibility in real-time scenarios.
### The proposed densely connected networks.
With a network that is densely connected, each layer's input receives data from all the preceding layers concatenated together. This approach offers a pair of major advantages. First, the vanishing gradient issue is avoided by the dense connections to each of the preceding layers. Second, it is found that a narrower dense network outperforms a broader normal network, thereby increasing the network's parameter efficiency.
            In the proposed approach, we suggest a dense block with dilated connections be applied after every layer in both the decoder and the encoder. There are five 2-dimensional convolutional layers in each dense block. The causal convolutions ensure that the suggested method is appropriate for real-time application. Therefore, the interframe convolutions are causal. After every convolution, parametric ReLU (PReLU) nonlinearity and layer normalization is applied. The dilation rates are set at 1, 2, 4, 8, and 16 in each dense block.
                                  
                 
### The proposed sub-pixel convolutions.
Within a convolutional neural network, sub-pixel convolutions, which are learnable, are employed as an up-sampling layer. In order to prevent checkerboard artifacts, subpixel convolution is used in this paper as a better replacement for transposed convolution. Firstly, an input speech signal is up sampled by adding zeros amid the subsequent samples in a transposed convolution, then a convolutional layer is used to generate a signal with non-zero elements. For a non-symmetric configuration that yields checkerboard artifacts, the filter length should be a factor of the filter stride. Sub-pixel convolution involves applying convolution to the original signal (no zeros are inserted) and multiplying the output channel count by the up-sampling rate to effectively increase the channel count.
            To obtain the necessary up sampled signal, the additional channels are reshaped. Figure 3 illustrates the up sampling of a 1-dimensional signal by a two-fold increase using sub-pixel convolution.


### The proposed model architecture.
The Figure 4 displays the architectural diagram of the model layout. The model is made up of the following layers: input, encoder, dense and dilated blocks, decoder, and output. Except for the output layer, all convolutions adhere to PReLU nonlinearity and layer normalization.
        The size of the model's input is [size of the batch, 1, number of frames, size of the frame]. Filters of size (1, 1) are used in the input layer to raise the total number of channels to 64. A dense block follows the input layer. Filters with a size of (2, 3) and 64 output channels are utilized for convolutions in all densely connected blocks.
           In the encoding phase of the proposed model, the dimension reduces initially at every layer (down sampling) along the frame axis by half while employing a convolution with filters of size (1, 3) and a stride of (1, 2). A dense block follows the down sampling.  The encoder's dense blocks at the end of each layer aid with context aggregation at various resolutions. The encoder has six layers of this type, and its ultimate output is [size of the batch, 64, number of frames, (size of the frame)/64].
            The size of the signal is gradually rebuilt to its original size by the decoder using dense blocks and sub-pixel convolutions. Each layer in the decoder receives its input data from the concatenation, which is in parallel with the channel axis, of the output from the layer before it and the output data from the equivalent symmetrical layer present in the encoder. The size of the input along its frame axis is doubled by sub-pixel convolutions using filters of size (1, 3). Lastly, the output layer outputs the enhanced frames by a single channel using filters of size (1, 1).
            To accommodate the time-varying nature of speech, we incorporate two GRU layers between the encoder and decoder. The frequency dimensions are expanded to match the input shape needed by the GRU. The encoder output's depth dimension is transformed into a series of feature vectors before being fed into the GRU layers. Following the GRU layers, the output sequence is reshaped to fit the decoder. GRU employs two gates, as opposed to LSTM's three gates, which reduces network complexity and enhances performance. Each layer effectively captures the temporal dynamics of speech.

 ### The proposed squeezed temporal convolutional modules.
The temporal convolutional modules (TCMs) have been extensively used recently in tasks involving speech distinction and target speaker identification. TCMs can achieve equivalent or even higher performance in time sequence modelling than LSTMs, and because they employ parallelable convolutions, less inference time is required. STCM, a lightweight TCN that reduces the number of parameters by squeezing the features into a smaller dimension with a dilated convolution, make up a STCN. To achieve a broad temporal receptive field, we employ STCN as the sequential module, which is applied after each dense block in the proposed model. Each STCN stacks 5 STCM units with an exponentially rising dilation rate d=1, 2, 4, 8, 16 as shown in Figure 5.
            The network can enhance speech information recovery by using the connection between several temporal scales. Figure 6 shows that STCM consists of three convolutions: an input 1x1 convolution, a gated depth-wise dilated convolution (GDDC), and an output 1x1 convolution. The input and output 1x1 convolutions are used to squeeze and restore the feature dimension, respectively, and GDDC differs from depth-wise dilated convolution in traditional TCM in three ways.
            First, the speech spectrum is time-frequency sparse, the DCC layer in the GDDC is less efficient at accurately capturing the information. In addition, GDDC adds a gating branch to let the gradient back-propagation process flow with information. In order to change the characteristic distribution of the main branch, the gating branch uses the sigmoid activation function to convert the output of DCC to the values (0, 1). To enhance network convergence, it is important to include layer normalization and PReLU layers between consecutive convolutional layers.
                                         
### The proposed loss functions.
We integrate two losses during the training of the model. Firstly, the overlap-and-add method is used to convert the enhanced frames into a waveform. To determine the loss at the level of utterances in the time domain, the mean squared error of the clean and enhanced utterances is used. The loss in the time-domain is expressed as: 
                                             l_t (h,h ̂) =  (1 )/l ∑_(n=0)^(l-1)▒〖〖(h_i [n] - h ̂_i  [n])〗^2  〗                            (1)
             L represents the length of utterance,            
             h[n] refers to the nth sample of the clean utterance and,
             h ̂[n] represents the nth sample of the enhanced utterance.
            Secondly, using l1 loss over the l1 norm of the STFT coefficients, we perform STFT on the utterances. The loss in the frequency domain is expressed as:
             l_f (h,h ̂) =  (1 )/(T·F) ∑_(t=1)^T▒〖∑_(f=1)^F▒〖 |[|〖H(x,y)〗_r | + |〖H(x,y)|〗_i]-[|〖H ̂(x,y)〗_r | + |〖H ̂(x,y)〗_i]| 〗  〗          (2)
              T represents total frame count,
              F represents the frequency bin count,       
              H (x, y) refers to the STFTs of h of T-F units,
            H ̂(x, y) represents the STFTs of h ̂ of T-F units and,
            The imaginary and real components of a complex variable H are represented as Hi and Hr, respectively
            Ultimately, the losses in the frequency and time domains are integrated as follows:
                             l(h,h ̂ )= (1 - β)* l_f (h,h ̂ )+ β * l_t (h,h ̂ )                                                          (3)
Where, β is a tailored hyper-parameter for the validation set.


##REFERENCES

[1] Ashutosh Pandey and DeLiang Wang1, "Densely connected neural network with diluted convolution for real-time speech enhancement in the time domain", 2022, 6629-6631.

[2] Huang, Z. Liu, L. Van Der Maaten, and K. Q. Weinberger, “Densely connected convolutional networks,” in IEEE conference on computer vision and pattern recognition, 2017, 4700–4708.

[3] K. Tan and D. Wang, “A convolutional recurrent neural network for real-time speech enhancement,” in Proc. Inter-speech, Hyderabad, India, 2018, pp. 3229–3233.
[4] Q. Wang et al., “Voice filter: Targeted voice separation by speaker conditioned spectrogram masking,” in Proc. Inter-speech, Graz, Austria, 2019, pp. 2728–2732. 
[5] A. Li, C. Zheng, C. Fan, R. Peng, and X. Li, “A Recursive network with dynamic attention for monaural speech enhancement,” in Proc. Inter-speech, Shangai, China, Oct. 2020, pp. 2422–2426
[6] A. Défossez, G. Synnaeve, and Y. Adi, “Real time speech enhancement in the waveform domain,” in Proc. Inter-speech, Shanghai, China, Oct. 2020, pp. 3291–3295.
[7] Ashutosh Pandey, DeLiang Wang, “Dense CNN With Self-Attention for Time-Domain
Speech Enhancement,” IEEE/ACM TRANSACTIONS ON AUDIO, SPEECH, AND LANGUAGE PROCESSING, VOL. 29, 2021
[8] Ullah, Rizwan, et al. "End-to-End Deep Convolutional Recurrent Models for Noise Robust Waveform Speech Enhancement.", 2022, pp.7782
[9] Wang, Kai, Bengbeng He, and Wei-Ping Zhu. "TSTNN: Two-stage transformer based neural network for speech enhancement in the time domain."  IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021.
[10] Irhum Shafkat, "Intuitively Understanding Convolutions for Deep Learning", 2018.

[11] Mukul Khanna, "DenseNet -Densely Connected Convolutional Networks", 2019.

[12] L. Ba, J. R. Kiros, and G. E. Hinton, “Layer normalization,” 2016.

[13] C.Jannu, S.D.Vanambathina, “Shuffle Attention U-Net for Speech Enhancement in Time Domain,” 2023

[14] K. He, X. Zhang, S. Ren, and J. Sun, “Delving deep into rectifiers: Surpassing human-level performance on imagenet classification”, 2015, 1026–1034.

[15] A. W. Rix, J. G. Beerends, M. P. Hollier, and A. P. Hekstra, “Perceptual evaluation of speech quality (PESQ) - a new method for speech quality assessment of telephone networks and codecs,” 2001, 749–752.

[16] Andong Li, Wenzhe Liu, Chengshi Zheng, and Xiaodong Li, “Two Heads are Better Than One: A Two-Stage Complex Spectral Mapping Approach for Monaural Speech Enhancement”, 2019, 1830-1832

[17] Andong Li, Chengshi Zheng, Minmin Yuan, Wenzhe Liu, Xiao Wang, Xiaodong Li, and Yi Chen, “A Neural Beamspace-Domain Filter for Real-Time Multi-Channel Speech Enhancement,” 2022, 7-8

[18] Jun Chen, Shimin Zhang, Shulin He, Shidong Shang, Tao Yu, Wei Rao, Weixin Zhu, Yannan Wang, and Yukai Ju, “TEA-PSE 3.0: Tencent-Ethereal-Audio-lab Personalized Speech Enhancement system for ICASSP 2023 DNS-challenge”, 2023

[19] K. Tan, J. Chen, and D. Wang, “Gated residual networks with dilated convolutions for monaural speech enhancement,” 2019, 189–198.

[20] C.H. Taal, R. C. Hendriks, R. Heusdens, and J. Jensen., “An algorithm for intelligibility prediction of time– frequency weighted noisy speech,” IEEE Transactions on Audio, Speech, and Language Processing, vol. 19, no. 7, pp. 2125–2136, 2011.
[21] W. Rix, J. G. Beerends, M. P. Hollier, and A. P. Hekstra., “Perceptual evaluation of speech quality (PESQ) - a new method for speech quality assessment of telephone networks and codecs,” In ICASSP, pp. 749–752, 2001.
[22] C. Valentini-Botinhao et al., “Noisy speech database for training speech enhancement algorithms and tts models,” 2017.
[23] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, “Librispeech: an asr corpus based on public domain audio books,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2015, pp. 5206–5210.
[24] Loizou, P. "NOIZEUS: A noisy speech corpus for evaluation of speech enhancement algorithms." Speech Communication 49 (2017): 588-601.
[25] Veaux C, Yamagishi J, King S. “The voice bank corpus: Design, collection and data analysis of a large regional accent speech database.” In 2013 International Conference Oriental COCOSDA held jointly with 2013 Conference on Asian Spoken Language Research and Evaluation(O-COCOSDA/CASLRE). IEEE, 2013: 1-4.
[26] Joachim Thiemann, Nobutaka Ito, and Emmanuel Vincent, “The diverse environments multi-channel acoustic noise database: A database of multichannel environ- mental noise recordings,” The Journal of the Acoustical Society of America, vol. 133, no. 5, pp. 3591–3591, 2013.
[27] D. P. Kingma and J. L. Ba, “Adam: A method for stochastic optimization,” in Proc. Int. Conf. Learn. Representations, San Diego, CA, USA, 2015, pp. 1–15.
[28] C. K. A. Reddy et al., “ICASSP 2021 deep noise suppression challenge,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process., Jun. 2021, pp. 6623–6627.
[29] A. Varga and H. J. Steeneken, “Assessment for automatic speech recognition: II. NOISEX-92: A database and an experiment to study the effect of additive noise on speech recognition systems,” Speech Commun., vol. 12, no. 3, pp. 247–251, Jul. 1993.
[30] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” in Advances in neural information processing systems, 2017, pp. 5998–6008. 

[31] A. Pandey and D. Wang, “Dense cnn with self-attention for time-domain speech enhancement,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, 2021, pp. 1270–1279.
