## Tasks

- CLI for training
- CLI for inference
- [DONE] CLI for help page
- CLI for evaluate
    - [DONE] eval on json only
    - [DONE] evalute on dataset vs json options
    - [DONE] add CLI flexibilty for diff image sizes
    - [DONE] add CLI for different mapping + explanation
    - have single eval produce results file
- CLI for multi-model evaluation
- determine tp_fp_fn function diff
- add type hints
- modify raitedataset object call to be flexible for other input parameters
- run pylint on all files and fix warnings (pylint)
- run formatter on all files (black)
- debug final_ap
- clean up comments
- clean up readme

Test for evaluation:
- different image sizes 400x400, 800x800, 400x800
- different mappings {0:0, 0:1}?
- 