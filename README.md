# Keyphrase Extraction using SciBERT (Semeval 2017, Task 10)

Deep Keyphrase extraction using SciBERT.

## Usage

1. Clone this repository and install `pytorch-pretrained-BERT`
2. From `scibert` repo, untar the weights (rename their weight dump file to `pytorch_model.bin`) and vocab file into a new folder `model`.
3. Change the parameters accordingly in `experiments/base_model/params.json`. We recommend keeping batch size of 4 and sequence length of 512, with 6 epochs, if GPU's VRAM is around 11 GB.
4. For training, run the command `python train.py --data_dir data/task1/ --bert_model_dir model/ --model_dir experiments/base_model`
5. For eval, run the command, `python evaluate.py --data_dir data/task1/ --bert_model_dir model/ --model_dir experiments/base_model --restore_file best`

## Results

### Subtask 1: Keyphrase Boundary Identification

We used IO format here. Unlike original SciBERT repo, we only use a simple linear layer on top of token embeddings.

On test set, we got:

1. **F1 score**: 0.6259
2. **Precision**: 0.5986
3. **Recall**: 0.6558
4. **Support**: 921

### Subtask 2: Keyphrase Classification

We used BIO format here. Overall F1 score was 0.4981 on test set.

|          | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| Process  | 0.4734    | 0.5207 | 0.4959   | 870     |
| Material | 0.4958    | 0.6617 | 0.5669   | 807     |
| Task     | 0.2125    | 0.2537 | 0.2313   | 201     |
| Avg      | 0.4551    | 0.5527 | 0.4981   | 1878    |

### Future Work

1. Some tokens have more than one annotations. We did not consider multi-label classification.
2. We only considered a linear layer on top of BERT embeddings. We need to see whether SciBERT + BiLSTM + CRF makes a difference.

## Credits

1. SciBERT: https://github.com/allenai/scibert
2. HuggingFace: https://github.com/huggingface/pytorch-pretrained-BERT
3. PyTorch NER: https://github.com/lemonhu/NER-BERT-pytorch
4. BERT: https://github.com/google-research/bert
