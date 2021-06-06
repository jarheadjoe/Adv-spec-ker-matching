# Adversarial Spectral Kernel Matching for Unsupervised Time Series Domain Adaptation

## Installation

### Datasets

Download data and convert to .tfrecord files for TensorFlow
(*./generate_tfrecords.sh*)

### Packages

We require the following packages.
Adjust for your computer setup.

    module load cuda/10.1.105 cudnn/7.6.4.38_cuda10.1 python3/3.7.4
    pip install --user --upgrade pip
    export PATH="$HOME/.local/bin:$PATH"
    pip3 install --user --upgrade pip
    pip3 install --user --upgrade numpy cython
    pip3 install --user --upgrade tensorflow-gpu pillow lxml jupyter matplotlib pandas scikit-learn scipy tensorboard rarfile tqdm pyyaml grpcio absl-py
    
    # If using --moving_average or F1 score metrics (typically tensorflow-addons, but that errors at the moment with TF 2.2)
    pip3 install --user git+https://github.com/tensorflow/addons.git@r0.9



## Usage

### Example

Train our model on person 14 of the UCI HAR dataset
and adapt to person 19.

    python3 main.py \
        --logdir=example-logs --modeldir=example-models \
        --method=smd --dataset=ucihar --sources=14 \
        --target=19 --uid=0 --debugnum=0 --gpumem=0

Then evaluate that model on the holdout test data, outputting the results to a
YAML file.

    mkdir -p results
    python3 main_eval.py \
        --logdir=example-logs --modeldir=example-models \
        --jobs=1 --gpus=1 --gpumem=0 \
        --match="ucihar-0-smd-[0-9]*" --selection="best_target" \
        --output_file=results/results_example_best_target-ucihar-0-daws.yaml

Note: there are a number of other models (e.g. ``--model=lstmfcn``), methods (e.g.
``--method=dann``), datasets (e.g. ``--dataset=wisdm_at``), etc. implemented that
you can experiment with beyond what was included in the paper.

### Analysis

Then look at the resulting *results/results_\*.yaml* files or analyze with
*analysis.py*.

In addition, there are scripts for visualizing the datasets
(*datasets/view_datasets.py*), viewing dataset statistics
(*dataset_statistics.py*), and displaying or plotting the class balance of the
data (*class_balance.py*, *class_balance_plot.py*).

## Navigating the Code

Here is an outline of the key elements of the code.

### Models

- ``--model=fcn`` base feature extractor in paper
- `--model=lstmfcn` fcn+lstm feature extractor

### Methods

- `dannMethods.py` includes methods based on domain classifier (DANN, CoDATs)
- `nannMethods.py` includes methods based on non-learnable metrics (DDC, DAN, CORAL, etc.)

- `nannMethods.py` includes methods based on learnable metrics (our method)