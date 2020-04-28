# TipSelectorAddition
An extension of the TipSelector algorithm using additional information tokens. 

Code for my iteration and additions to the TipSelector algorithm, using n-grams (up to 4) and (negator) POS-bigrams as information tokens.

Software is written in PYTHON Version 3.7.

# Installation Instructions:
**Downloads**
- TipSelector Dataset: https://tinyurl.com/TipSelectorData
- Download the Pycharm IDE (other IDE's may work but require additional configuration) from https://www.jetbrains.com/pycharm/

**Setup of the Environment**
1. Download and unzip the TipSelectorAddition.zip
2. Place the folder TipSelectorAddition in your Pycharm project folder and import as a new project
3. Install all packages listed in the packages.csv, installation method may vary per package (This should work without installing if you are directly using the Pycharm IDE for most packages)
4. Open the py files mentioned in code.txt inside Pycharm. They are located in ...\TipSelectorAddition\venv\Scripts\
5. Place the folders parsed and landing_pages inside the TipSelectorData.zip in your current working directory (or configure your own working directory)
6. Change the working directory in the code files to the one you are using

# Software explanation:
- LDA_variables.py: Grabs the topic distributions used for the LDA in the similarity function.
- similarity function.py: Calculates all sets of 5 hotels most similar.
- First code (opening_tokens).py: Generate all information tokens and count the frequency they occur.
- Fisher's test.py: Running the Fisher's test in order to select information tokens based on relative frequency
- Tip selection.py: Selects tips based on those sentences which cover the most tokens
- random sentences.py: Selects random sentences from the dataset for the user-study
