# 🔥 xLSTM for Intelligent Fault Diagnosis of Rolling Bearings

The pytorch implementation of xLSTM for Intelligent Fault Diagnosis of Rolling Bearings. **This is just an experimental report!** 

###                                                                           The training speed is particularly slow!
### This is just a very basic report!





## Brief introduction  
Experimental report on using xLSTM for fault diagnosis. Replace the BiLSTM module in DCA-BiGRU with the module in xLSTM.

## Result

- Link:https://caiyun.139.com/m/i?085Cta92jb6QO Code:7C8c


- Verification set result report

  |  Block  | Performance                |
  | :-----: | -------------------------- |
  | BiLSTM  | 96.26%                     |
  | BisLSTM | 95.15%                     |
  |  LSTM   | 56.42%                     |
  |  mLSTM  | 10.04%                     |
  | s_mLSTM | 92.13%                     |
  |  sLSTM  | 96.65% ($\uparrow 0.39$ %) |

## Citation



```html
@article{beck2024xlstm,
  title={xLSTM: Extended Long Short-Term Memory},
  author={Beck, Maximilian and P{\"o}ppel, Korbinian and Spanring, Markus and Auer, Andreas and Prudnikova, Oleksandra and Kopp, Michael and Klambauer, G{\"u}nter and Brandstetter, Johannes and Hochreiter, Sepp},
  journal={arXiv preprint arXiv:2405.04517},
  year={2024}
}
```
```html
@article{he2023physics,  
title = {Physics-informed interpretable wavelet weight initialization and balanced dynamic adaptive threshold for intelligent fault diagnosis of rolling bearings},  
journal = {Journal of Manufacturing Systems},  
volume = {70},  
pages = {579-592},  
year = {2023}, 
doi = {10.1016/j.jmsy.2023.08.014},  
author = {Chao He and Hongmei Shi and Jin Si and Jianbo Li}
```
```
@article{he2024interpretable,
  title={Interpretable physics-informed domain adaptation paradigm for cross-machine transfer diagnosis},
  author={He, Chao and Shi, Hongmei and Liu, Xiaorong and Li, Jianbo},
  journal={Knowledge-Based Systems},
  pages={111499},
  year={2024},
  doi = {10.1016/j.knosys.2024.111499}
}
```
```
@article{he2023idsn,
  title={IDSN: A one-stage interpretable and differentiable STFT domain adaptation network for traction motor of high-speed trains cross-machine diagnosis},
  author={He, Chao and Shi, Hongmei and Li, Jianbo},
  journal={Mechanical Systems and Signal Processing},
  volume={205},
  pages={110846},
  year={2023},
  doi = {10.1016/j.ymssp.2023.110846} 
}
```
```
@article{He2024InterpretableMD,
  title={Interpretable modulated differentiable STFT and physics-informed balanced spectrum metric for freight train wheelset bearing cross-machine transfer fault diagnosis under speed fluctuations},
  author={He, Chao and Shi, Hongmei and Li, Ruixin and Li, Jianbo and Yu, ZuJun},
  journal={Advanced Engineering Informatics},
  volume={62},
  pages={102568},
  year={2024},
  doi = {10.1016/j.aei.2024.102568} 
}
```




  

## References

- The first GitHub repositories that implement xLSTM are:
  
  - https://github.com/andrewgcodes/xlstm [![GitHub Repo stars](https://camo.githubusercontent.com/c0fd25e1080a7fabb1399b4a8e777d82cfbdbfeebd90c3bb2ee5d9c6348a195a/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f616e6472657767636f6465732f786c73746d3f7374796c653d736f6369616c)](https://camo.githubusercontent.com/c0fd25e1080a7fabb1399b4a8e777d82cfbdbfeebd90c3bb2ee5d9c6348a195a/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f616e6472657767636f6465732f786c73746d3f7374796c653d736f6369616c)
  - https://github.com/muditbhargava66/PyxLSTM [![GitHub Repo stars](https://camo.githubusercontent.com/304a28e90243b07a7de5db9b94f152455574d95e41b33503a9cef9eb76d82648/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f6d75646974626861726761766136362f5079784c53544d3f7374796c653d736f6369616c)](https://camo.githubusercontent.com/304a28e90243b07a7de5db9b94f152455574d95e41b33503a9cef9eb76d82648/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f6d75646974626861726761766136362f5079784c53544d3f7374796c653d736f6369616c)
  - https://github.com/kyegomez/xLSTM [![GitHub Repo stars](https://camo.githubusercontent.com/f1ac72593ea051e4afa6946525a90e8d72b4cb75cd64d360c4f077645734f16e/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f6b7965676f6d657a2f784c53544d3f7374796c653d736f6369616c)](https://camo.githubusercontent.com/f1ac72593ea051e4afa6946525a90e8d72b4cb75cd64d360c4f077645734f16e/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f6b7965676f6d657a2f784c53544d3f7374796c653d736f6369616c)
  - https://github.com/akaashdash/xlstm [![GitHub Repo stars](https://camo.githubusercontent.com/e64d1727af6c8fe1b6f66fd2d425034c2b8e2e5c312292826676a95af59f4989/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f616b61617368646173682f786c73746d3f7374796c653d736f6369616c)](https://camo.githubusercontent.com/e64d1727af6c8fe1b6f66fd2d425034c2b8e2e5c312292826676a95af59f4989/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f616b61617368646173682f786c73746d3f7374796c653d736f6369616c)
  - https://github.com/myscience/x-lstm [![GitHub Repo stars](https://camo.githubusercontent.com/26de699caf7d30cf1d650eb97acb147e57bf318c2c8a125f81acc996f26fee12/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f6d79736369656e63652f782d6c73746d3f7374796c653d736f6369616c)](https://camo.githubusercontent.com/26de699caf7d30cf1d650eb97acb147e57bf318c2c8a125f81acc996f26fee12/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f6d79736369656e63652f782d6c73746d3f7374796c653d736f6369616c)
  
- ​    https://github.com/AI-Guru/xlstm-resources  
  ​    
  
  ## Contact
  
  - **Chao He**
  - **chaohe#bjtu.edu.cn (please replace # by @)**
  
  ​      
