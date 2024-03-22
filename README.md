# IEEE ICME 2024 GRAND CHALLENGE - Semi-supervised Acoustic Scene Classification under Domain Shift
## SEMI-SUPERVISED ACOUSTIC SCENE CLASSIFICATION UNDER DOMAIN SHIFT USING AN ATTENTION MODULE AND ANGULAR LOSS
Michael Neri and Marco Carli
Roma Tre University, Department of Industrial, Electronic, and Mechanical Engineering

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


