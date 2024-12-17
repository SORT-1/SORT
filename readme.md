# Artifact for the Paper: "Subgraph-Oriented Testing for Deep Learning Libraries"

This artifact provides the replication package and details of the survey of the paper *"Subgraph-Oriented Testing for Deep Learning Libraries"*.

## This artifact contains:
- [Introduction](#introduction)
- [Code](#code)
- [Results](#results)
- [Survey](#survey)

## Introduction

**SORT** (**S**ubgraph-**O**riented **R**ealistic **T**esting) is a subgraph-oriented DL library testing method. It takes popular API interaction patterns, represented as frequent subgraphs of model computation graphs, as test subjects. Moreover, SORT prepares test inputs by referring to features of runtime inputs for each API in executing real-life benchmark data.

## Code

The [**`/code`**](code) folder contains the scripts to reproduce the experiments.

### Folder Structure
```
code
├── requirements.txt
├── run_testing.py
├── configs/
│   ├── input_feature_example.json
│   ├── subgraphs_example.json
│   └── API_def_example.txt
└── utils/
    ├── ErrorStartException.py
    ├── invoke_str_generator.py
    ├── subgraph_util.py
    ├── torch_input_generator.py
    └── __init__.py
```

### Create Python Environment

```bash
conda create -n subgraph_testing python=3.8
conda activate subgraph_testing
pip install -r requirements.txt
```

### Config Files
- Configuration files in [**`/code/configs`**](code/configs) contain the API input features and frequent subgraphs for the experiments.
    - **`input_feature_example.json`**: The input features of APIs.
    - **`subgraphs_example.json`**: The frequent subgraphs.
    - **`API_def_example.txt`**: Definition of APIs for PyTorch. 
- How to collect the config files:
  - **Input features**: These can be collected using any instrumentation method. 
  - **Frequent subgraphs**:
    - Collect the computation graph of the model to be tested using [torchview](https://github.com/mert-kurttutan/torchview). 
    - Extract frequent subgraphs using [gSpan](https://github.com/betterenvi/gSpan).

### Key Scripts:
- **`run_testing.py`**: This is the main script for running the tests. It executes subgraph testing based on the configurations provided in the `configs` folder.

### Running the Tests
To reproduce the experiment results from the paper, run the following commands:

1. **Run the Subgraph Testing**:

    Command Parameters for `python run_testing`

   - **`-b`**: Number of batches for generating test cases  
   - **`-bs`**: Number of test cases generated per batch  
   - **`-s`**: File path for the subgraph being tested  
   - **`-config`**: File path for the API input feature configuration of the subgraph being tested   
   - **`-d`**: File path to save the input-output results of the test cases  
   - **`-def`**: API definition file.

    **Example**
    ```sh
    python run_testing.py -b 1 -bs 5 -s ./configs/subgraphs_example.json -config ./configs/input_feature_example.json -d ./results/input_output_data -def ./configs/API_def_example.txt
    ```

2. **Results**:
   The output of APIs in the subgraphs will be saved in the `results/input_output_data` file.

## Results

- In [**`/results`**](/results), We provide the subgraphs we tested and the testing results.

## Survey

- In [**`/survey/survey.md`**](survey/survey.md), we provide the details of the survey conducted to understand the perspectives of developers in testing deep learning libraries.
  

## Contact

If you have any questions with the artifact, please reach out to [yansong@whu.edu.cn](mailto:yansong@whu.edu.cn).

