# RBE 577 Hw 3

An Implementation of a Recurrent Neural Network to Predict the Trajectory of a Dubins Airplane

The similar dataset to that used in this implementation can be generated using the test_dubinEHF3d method found in this repository. However, before running the code, you must first ensure your MATLAB path is in the same folder as this repository and that a new folder named "data" is instantiated.

The Dubins airplane trajectory prediction method was written in python 3.10.9 and can be found in the attached rnn_text.py file. To run the file, first create a virtual environment that runs python version 3.10.9. Then, ensure the virtual environment contains the following packages:

Package:               Version:
---------------------   --------------
absl-py                 2.1.0
aiohappyeyeballs        2.4.3
aiohttp                 3.10.10
aiosignal               1.3.1
async-timeout           4.0.3
certifi                 2024.8.30
charset-normalizer      3.4.0
contourpy               1.3.0
cycler                  0.12.1
datasets                3.0.1
dill                    0.3.8
filelock                3.16.1
fonttools               4.54.1
frozenlist              1.4.1
fsspec                  2024.6.1
grpcio                  1.67.0
huggingface-hub         0.26.0
idna                    3.10
Jinja2                  3.1.4
kiwisolver              1.4.7
Markdown                3.7
MarkupSafe              3.0.2
matplotlib              3.9.2
mpmath                  1.3.0
multidict               6.1.0
multiprocess            0.70.16
networkx                3.4.1
numpy                   2.1.2
packaging               24.1
pandas                  2.2.3
pillow                  11.0.0
pip                     24.2
propache                0.2.0
protobuf                5.28.2
pyarrow                 17.0.0
pyparsing               3.2.0
python-dateutil         2.9.0.post0
pytz                    2024.2
PyYAML                  6.0.2
requests                2.32.3
scipy                   1.14.1
setuptools              74.1.2
six                     1.16.0
sympy                   1.13.1
tensorboard             2.18.0
tensorboard-data-server 0.7.2
torch                   2.5.0
torch-tb-profiler       0.4.3
torchtext               0.18.0
torchvision             0.20.0
tqdm                    4.66.5
typing_extensions       4.12.2
tzdata                  2024.2
urllib3                 2.2.3
Werkzeug                3.0.4
wheel                   0.44.0
xlrd                    2.0.1
xxhash                  3.5.0
yarl                    1.15.5

Ensure that the IDE you are using is running the python interpreter located within the virtual environment you created to allow the packages to be recognized.

The file outputs 10 figures which are images of the normalized ground truth trajectories versus the trajectory predictions. The images are saved to a folder titles "images."

Graphs of the loss and accuracy in the training and validation data as a function of epoch can be seen while the code is running and/or after running has completed by instantiating the TensorBoard.

Python Version : Python 3.10.9