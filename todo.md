## Tasks

### Priority
- add wandb to training setup for RAITE - refactor training code to run w wandb ******
    - [DONE] succesful training with wandb
    - clean wandb workflow - ensure useful metric are being tracked
    - save models to model repo in wandb
- CLI for training
- CLI for inference/ visualization

### Not Priority
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
- add batch training
- modify raitedataset object call to be flexible for other input parameters
- run pylint on all files and fix warnings (pylint)
- run formatter on all files (black)
- debug final_ap
- clean up comments
- clean up readme

Tests for evaluation:
- different image sizes 400x400, 800x800, 400x800
- different mappings {0:0, 0:1}?
- 

### Later Priority
- connect with isaacsim