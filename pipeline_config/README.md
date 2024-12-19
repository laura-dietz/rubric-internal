# Pipeline Configuration Files

The motivation for pipeline configuration files is to be able to easily make changes to RUBRIC experimental settings in a single location (the config file). Currently, these files only support configuration of the loading and running of HuggingFace models:

1. `hf_model_init`: Arguments to be passed to the `AutoModel.from_pretrained(...)` call that constructs the HuggingFace model that powers the pipeline.
1. `hf_tokenizer_init`: Arguments to be passed to the `AutoTokenizer.from_pretrained(...)` call that constructs the HuggingFace tokenizer used by the pipeline.
1. `hf_pipeline_init`: Arguments to be passed to the HuggingFace pipeline constructor
1. `hf_pipeline_call`: Arguments to be passed when invoking the HuggingFace pipeline's `__call__(...)` method.

For each of these things, some RUBRIC-specific default arguments are set depending on the type of pipeline being used (see `t5_qa.py`). Arguments provided in the config files always override the defaults. A config file can be passed to the `exam_grading.py` script via the `--pipeline-config-file` command line argument.
