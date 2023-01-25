# Vectorizer Robustness

Submission code for the paper [On the Robustness of Text Vectorizers]()

## General Organization

The scripts producing the experiments are in the main directory. One first has
to set config.ini. Then run train_model.py to train / calibrate the models, and
run influence scripts to launch the actual experiments. Finally, the plots are
obtained by running plot scripts.

 * `requirements.txt`: versions of the different librairies used
 * `config.ini`: indicate the path to the main directory here
 * `train_model.py`: launch this script to train / calibrate the different
   vectorizers
 * `influence_length_document.py`: creates data for the document length experiment
 * `influence_number_replacements.py`: creates data for the number of replacements
   experiments
 * `plot_influence_length_document.py`: script used to produce Figure 1, 5, and 7
 * `plot_influence_number_replacements.py`: script used to produce Figure 2, 6,
   and 8
 * `norm_original_embedding.py`: script used to produce Figure 9
 
## Generating the figures
 
In order to generate the figures, you should install the requirements, in
particular `pytorch`, `torchtext` and `gensim`.

    $ pip install -r requirements.txt
    
Then, you should create `config.ini` by modifying `config.ini.example`

    $ cp config.ini.example config.ini
    
Set up `root_path` to a suitable directory. You should then train the models

    $ python train_model.py
    
It should download the necessary data. The script `influence_length_document.py`
and `influence_number_replacements.py` will pickle the dat for the experiments in
Figures 1, 2, 5, 6, and 7. Plotting is performed by preprending `plot_` to these
files.
