# TreeTransitions
Research code to test hypotheses about recurrent neural networks learning hierarchically structured transition probabilities 

## Usage

To run the default configuration, call `treetransitions.job.main` like so:

```python
from treetransitions.job import main
from treetransitions.params import param2default

main(param2default)  # runs the experiment in default configuration
```

## Compatibility

Developed on Ubuntu 16.04 using Python3.7