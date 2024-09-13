# IEEE ICME 2024 GRAND CHALLENGE - Semi-supervised Acoustic Scene Classification under Domain Shift
## SEMI-SUPERVISED ACOUSTIC SCENE CLASSIFICATION UNDER DOMAIN SHIFT USING AN ATTENTION MODULE AND ANGULAR LOSS - RM3Team Submission

Michael Neri and Marco Carli

Department of Industrial, Electronic, and Mechanical Engineering, Roma Tre University, Rome, Italy

# Instructions
The structure of the project should be like the following:
- 📁 dev
    - 🎧 40c8570776704268be56bc0d1fa8a16f60_1.wav
    - 🎧 (...).wav (all other audio recordings of the development dataset)
- 📁 eval
    - 🎧 00b514a3c774405fb4005ea8c40d053e50_1.wav
    - 🎧 (...).wav (all other audio recordings of the evaluation dataset)
- 📁 TUT 2020 UAS 
    - 📁 audio
    - 🔢 meta.csv
- 📄 data.py (Dataset & Dataloaders)
- 📄 model.py (proposed approach)
- 📄 mobilenet.py (MobileFaceNet classifiers)
- 📄 utils.py 
- 📓 submission.ipynb
- 🔢 ICME2024_ASC_dev_label.csv
- 🔢 ICME2024_ASC_eval.csv
- 🧠 pretrained_TAU_pp5g.ckpt (weights after TAU pretraining)
- 🧠 pretrained_GC_hl19.ckpt (weights after multi-iteration FT process)
- 🧠 pretrained_GC_final_o2c4.ckpt (final weights after FT with unlabelled data)

Then, it is possible to run the code from 📓 submission.ipynb in Google Colab. 

-------------

If you use parts of this repository (mostly the attention module), please cite the following work:

```
@ARTICLE{Neri_TASLP_2024,
  author={Neri, M. and Archontis, P. and Krause, D. and Carli, M. and Virtanen, T.},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={{Speaker Distance Estimation in Enclosures from Single-Channel Audio}}, 
  year={2024},
  volume={},
  number={},
  pages={},
  doi={10.1109/TASLP.2024.3382504}
  }

@INPROCEEDINGS{Neri_ICMEW_2024,
  author={Neri, M. and Carli, M.},
  booktitle={IEEE International Conference on Multimedia and Expo Workshop (ICMEW)}, 
  title={{Semi-Supervised Acoustic Scene Classification under Domain Shift using Attention-based Separable Convolutions and Angular Loss}}, 
  year={2024},
  volume={},
  number={},
  pages={},
  doi= {10.1109/ICMEW63481.2024.10645482}}
```


