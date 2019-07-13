# Deep Convolutional Generative Adversarial Networks(DCGAN)
DCGANの基本的な型  
----
## 概要
用意した学習データからそれに似たデータ(64×64)を生成する。  

## version
- python 3.7.3
- torch 1.1.0  
- torchvision 0.3.0  

## 使用方法  
用意する学習データの数(x)によってbatch_size(bs)を変える必要がある。  
(x % bs == 0)  
## 使用例  
309枚のイラストを用意し、batch_size=103として実験を行った。  
学習過程の一部をgifファイルとして載せている(train_process.gif)
----


