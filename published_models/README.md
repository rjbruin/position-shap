# Published models

This package implements interfaces to all public models analyzed in the paper.

## Implementing a new model

- In a new file, implement a subclass of `base.py:BasePublishedModel`, adhering
  to the prescribed interface;
- Make sure the model you implement applies `required_grad_(True)` to the
  attribution sources (image input, position embeddings and all bias terms),
  which is necessary for the sensitivity attribution analysis to work;
- Add your new class to the `factory.py:get()` function (for the
  training/evaluation script to initialize the model correctly) and the
  `factory.py:IMPLEMENTATIONS` variable (for the testcases to discover all
  available models).
- In the main directory of the repository, run the testcases: `pytest
  published_models/test_all_implementations.py` and make sure there are no
  failed testcases;
- Commit your changes to a new branch and make a pull request. NOTE: please
  don't directly push to the main branch.