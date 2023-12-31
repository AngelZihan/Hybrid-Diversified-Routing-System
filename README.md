# Hybrid-Diversied-Routing-System

The Server-End code is used to start the sever, please run this code first

## Server-End Running

Two steps to run the sever

The first setp is to config the environment.

Then use the code to stat the server:

```sh
$ python Server.py
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

The client related files have been uploaded in: https://www.dropbox.com/scl/fo/a8o1dihgk0rrngrml8kww/h?rlkey=qkyrpqbtgahc3whmcnln7l6p2&dl=0, please make sure the processed.zip and COL_processed.zip be decompressed and be put into the same folder with the model-training folder put other files into the Client-End folder to make sure the code can run successfully.
## Change Parameter
| Parameter            | File      | How to Change                            |
|----------------------|-----------|------------------------------------------|
| Road File            | main.cpp  | Change the 'filename' address            |
| Query File           | main.cpp  | Change the 'queryFilename' address       |
| Algorithm            | main.cpp  | Uncomment the algorithm                  |
| Path Number          | main.cpp  | Change the 'k' value                     |
| Similarity Threshold | main.cpp  | Change the 't' value                     |
| model                | Server.py | Change the 'model' value to related file |

## Running Result

The running result shows the server answer 0-simple or 1-complex and the corresponding algorithms' running result:
running time and the length of each path.
