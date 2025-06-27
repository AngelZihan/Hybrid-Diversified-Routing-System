# Hybrid-Diversied-Routing-System

The Server-End code is used to start the sever, please run this code first

## Server-End Running

Two steps to run the sever

The first setp is to config the environment.

Then use the code to stat the server for NY or COL map:

```sh
$ python NYServer.py
```
or

```sh
$ python COLServer.py
```


The Client-End code is used to send request to the server and receive the answer

There are two exact algorithms and three heuristic algorithms in this repository:

| Algorithm          | Description                                                                                               |
|--------------------|-----------------------------------------------------------------------------------------------------------|
| onePass            | Implementation of the 'Finding k-shortest paths with limited overlap' exact algorithm (VLDBJ 2020)        |
| DKSP               | Implementation of the 'Diversified Top-k Route Plainning in Road Network' exact algorithm (VLDB 2022)     |
| Dynamic Similarity | Implementation of the 'Diversified Top-k Route Plainning in Road Network' heuristic algorithm (VLDB 2022) |
| ESX-C              | Implementation of the 'Finding k-shortest paths with limited overlap' heuristic algorithm (VLDBJ 2020)    |
| SVP-C              | Implementation of the 'Finding k-shortest paths with limited overlap' heuristic algorithm (VLDBJ 2020)    |

## Client-End Running

plz run this code after the server start

Two steps to run these algorithms:

The first step is to compile: 

```sh
$ make
```

Then, use the code to run:

```sh
$ ./classification
```
To run the NY map on the client side, please ensure that the NY-related file "NY_Result_changeKSP_fileNew_addKSPLost" is used in main.cpp.

To run the COL map on the client side, please ensure that the COL-related file "COL_Result_changeKSP_fileNew_addKSPLost2" is used in main.cpp.

## Change Parameter
| Parameter            | File                     | How to Change                            |
|----------------------|--------------------------|------------------------------------------|
| Road File            | main.cpp                 | Change the 'filename' address            |
| Query File           | main.cpp                 | Change the 'queryFilename' address       |
| Algorithm            | main.cpp                 | Uncomment the algorithm                  |
| Path Number          | main.cpp                 | Change the 'k' value                     |
| Similarity Threshold | main.cpp                 | Change the 't' value                     |
| Model                | NYServer.py/COLServer.py | Change the 'model' value to related file |


## Model Training
If you want to train the model by yourself. You can use the code to train the model for NY or COL map:

```sh
$ python NY_model_training.py
```
or
```sh
$ python COL_model_training.py
```
To train the NY model, please ensure that the NY-related files "NY_NodePair", "NY_additionalDataset", and "NY_processed.zip" are placed in the same directory as the Python script "NY_model_training.py". Note that "NY_processed.zip" must be decompressed before training.

To train the COL model, please ensure that the NY-related files "COL_NodePair", "NY_additionalDataset", and "NY_processed.zip" are placed in the same directory as the Python script "COL_model_training.py". Note that "COL_processed.zip" must be decompressed before training.

Due to GitHub’s file size limitations, all large files have been uploaded separately to Dropbox: https://www.dropbox.com/scl/fo/a8o1dihgk0rrngrml8kww/h?rlkey=qkyrpqbtgahc3whmcnln7l6p2&dl=0.
## Running Result

The running result shows the server answer 0-simple or 1-complex and the corresponding algorithms' running result:
running time and the length of each path.
