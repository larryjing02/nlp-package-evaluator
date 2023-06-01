# nlp-package-evaluator
Evaluates Stanford NLP's Stanza package on Universal Dependencies Treebank data, provided in CoNLL-U format.

To run: `python3 eval_f1.py`  
  
Ensure treebank data is provided in data directory  
  
This script compares POS tags with UD treebank data, calculating an aggregated F1 score by taking frequency-weighted average of individual F1 scores for each POS tag. Tracks individual binary confusion matrices across different tags.
