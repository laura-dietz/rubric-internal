import argparse
import asyncio

import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Tuple, List, Dict, Callable, NewType, Optional, Iterable
import typing
import openai
import torch
from transformers import pipeline, T5ForConditionalGeneration, GPT2TokenizerFast, T5TokenizerFast, T5Tokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, PretrainedConfig,AutoModelForQuestionAnswering,AutoTokenizer

from json import JSONDecodeError
from enum import Enum

import transformers
from . import openai_interface
from .openai_interface import query_gpt_batch_with_rate_limiting, OpenAIRateLimiter, FetchGptJson


from .test_bank_prompts import Prompt, QuestionPromptWithChoices,QuestionPrompt
from .batched_worker import BatchedWorker

DEFAULT_MAX_INPUT_TOKENS = 512
DEFAULT_MAX_OUTPUT_TOKENS = 512
DEFAULT_QUESTION_BATCH_SIZE = 100

os.environ["DSP_NOTEBOOK_CACHEDIR"] = str((Path(".") / "cache").resolve())

PromptGenerator = Callable[[Prompt],str]
PromptGeneratorQC = Callable[[Prompt],Dict[str,str]]

class ConfigKey(Enum):
    """top-level keys for JSON pipeline config files"""
    HF_MODEL_INIT = "hf_model_init"
    HF_TOKENIZER_INIT = "hf_tokenizer_init"
    HF_PIPELINE_INIT = "hf_pipeline_init"
    HF_PIPELINE_CALL = "hf_pipeline_call"
    RUBRIC_PIPELINE_INIT = "rubric_pipeline_init"

PipelineConfig = Dict[ConfigKey, Dict[str, Any]]

 
def load_pipeline_config(config_file: Optional[str] = None) -> PipelineConfig:
    # Load pipeline config from file
    # NOTE: OpenAI and VLLM pipelines currently only support tokenizer_init
    pipeline_config = {key: {} for key in ConfigKey}
    if config_file is not None:
        with open(config_file) as f:
            print(f"Loading pipeline config from {config_file}:")
            raw_config = json.load(f)
            for key in ConfigKey:
                if key.value in raw_config:
                    pipeline_config[key] = raw_config[key.value]
            unused_keys = set(raw_config.keys()) - set([key.value for key in ConfigKey])
            if unused_keys:
                print(f"WARNING: Unused keys in pipeline config: {unused_keys}")

    print("----------------------------------")
    print("Pipeline initialization overrides:")
    print("----------------------------------")
    for k,v in pipeline_config[ConfigKey.HF_PIPELINE_INIT].items():
        print(f"- {k}: {v}")
    print()

    print("---------------------------")
    print("Pipeline calling overrides:")
    print("---------------------------")
    for k,v in pipeline_config[ConfigKey.HF_PIPELINE_CALL].items():
        print(f"- {k}: {v}")
    print()

    print("-------------------------------")
    print("Model initialization overrides:")
    print("-------------------------------")
    for k,v in pipeline_config[ConfigKey.HF_MODEL_INIT].items():
        print(f"- {k}: {v}")
    print()

    print("-----------------------------------")
    print("Tokenizer initialization overrides:")
    print("-----------------------------------")
    for k,v in pipeline_config[ConfigKey.HF_TOKENIZER_INIT].items():
        print(f"- {k}: {v}")
    print()
    
    return pipeline_config


def create_gpt_client()->openai.OpenAI:
    return openai_interface.createOpenAIClient()


def create_vllm_client(base_url:str|None=os.getenv('VLLM_URL'))->openai.OpenAI:
    if base_url is None and os.getenv('VLLM_URL') is None:
        raise RuntimeError ("Must set environment variable \"VLLM_URL\". For localhost use \'http://[::0]:8000/v1\' ")

    return openai_interface.createOpenAIClient(api_key="NONE", base_url=base_url)

class FetchGptGrade(FetchGptJson):
    def __init__(self, gpt_model:str, max_tokens:int, client:openai.OpenAI, use_chat_protocol:True):
        super().__init__(gpt_model=gpt_model, max_tokens=max_tokens, client=client, use_chat_protocol=use_chat_protocol)


        json_instruction= r'''
Give the response in the following JSON format:
```json
{ "grade": int }
```'''
        self.set_json_instruction(json_instruction, field_name="grade")



def computeMaxBatchSize(modelConfig:PretrainedConfig)-> int:
    '''Estimates the batch size possible with a given model and given GPU memory constraints'''
    # TODO: make this its own script


    gpu_memory = 45634    # A40
    # Constants
    memory_for_activations_mib = gpu_memory / 2  # Half of the total GPU memory
    d_model = modelConfig.d_model  # 1024 Model dimension
    token_length = 200   # 512 Maximum token length
    bytes_per_parameter = 4  # FP32 precision

    # Calculating the maximum batch size
    # Memory required per token in a batch (in MiB)
    memory_per_token_mib = d_model**2 * bytes_per_parameter / (1024**2)

    # Total memory required for one batch of size 1
    total_memory_per_batch_mib = token_length * memory_per_token_mib

    # Estimate the maximum batch size
    max_batch_size = memory_for_activations_mib / total_memory_per_batch_mib
    return math.floor(max_batch_size)



from enum import Enum, auto
from abc import ABC, abstractmethod

class LlmEngine(Enum):
    HF_TF = auto()
    HF_TF_ASYNC = auto()
    OPEN_AI = auto()
    VLLM = auto()

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(arg:str):
        try:
            return LlmEngine[arg.upper()]
        except KeyError:
            raise argparse.ArgumentTypeError("Invalid llm-engine choice: %s" % arg)

        
class HfPipeline(Enum):

    def __str__(self):
        return self.name
    
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj
    
    def __init__(self, default_init_args):
        self.default_init_args = default_init_args

    @staticmethod
    def from_string(arg:str):
        try:
            return HfPipeline[arg.upper()]
        except KeyError:
            raise argparse.ArgumentTypeError("Invalid HfPipeline choice: %s" % arg)

    # Each pipeline type has a set of (overridable)
    # default parameters that are passed to the pipeline constructor
    text2text = {"use_fast": True, "batch_size": 1, "device": "0"}
    textgeneration = {"use_fast": True, "batch_size": 1, "device": "0"}
    # NOTE: you will need to pass a valid HuggingFace API token ("token")
    # in order to use the llama pipeline
    llama = {"token": None, "batch_size": 1, "device_map": "auto", "model_kwargs": {"torch_dtype": torch.bfloat16, "quantization_config": {"load_in_4bit": True}}}
    qa = {"use_fast": True, "batch_size": 1, "device": "0"}


       
class PromptRunner(ABC):
    @abstractmethod
    async def run_prompts(self, prompts: List[Prompt], context:str, pipeline_config: PipelineConfig = {}) -> List[str]:
        pass
    

    @abstractmethod
    async def call_pipeline(self, prompts: List[str], pipeline_config: PipelineConfig = {}) -> List[str]:
        pass
    
    @abstractmethod
    def get_tokenizer(self)-> AutoTokenizer:
        pass

    @abstractmethod
    def finish(self):
        pass

    def batchChunker(self, iterable):
        iterator = iter(iterable)
        while True:
            batch = list(itertools.islice(iterator, self.question_batch_size))
            if not batch or len(batch)<1:
                break
            yield batch



class HfTransformersQaPromptRunner(PromptRunner):
    def __init__(self, pipeline:transformers.Pipeline, tokenizer:AutoTokenizer, max_input_tokens: int, question_batch_size: int):
        self.hf_pipeline:transformers.Pipeline =pipeline
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens
        self.question_batch_size = question_batch_size
        self.default_hf_pipeline_call_args = {
            "num_beams": 5,
            "early_stopping": True,
            "max_length": self.max_input_tokens
        }

    async def run_prompts(self, prompts: List[Prompt], context:str, pipeline_config: PipelineConfig = {}) -> List[str]:
        converted_prompts = [prompt.generate_prompt_with_context_QC_no_choices(context=context, model_tokenizer=self.tokenizer, max_token_len=self.max_input_tokens) for prompt in prompts]
        return await self.call_dict_pipeline(dict_prompts=converted_prompts, pipeline_config=pipeline_config)


    async def call_dict_pipeline(self, dict_prompts: List[Dict[str,str]], pipeline_config: PipelineConfig = {}) -> List[str]:

        def processBatch(prompts, kwargs):
            resps = self.hf_pipeline(prompts, **kwargs)
            return [resp['answer'] for resp in resps]

        # override defaults with any explicitly provided arguments
        kwargs = self.default_hf_pipeline_call_args.copy()
        for k,v in pipeline_config[ConfigKey.HF_PIPELINE_CALL].items():
            kwargs[k] = v

        return list(itertools.chain.from_iterable(
                        (processBatch(batch, kwargs) for batch in self.batchChunker(dict_prompts)) 
                        )) 

    async def call_pipeline(self, prompts: List[str], pipeline_config: PipelineConfig = {}) -> List[str]:
        raise RuntimeError("QA pipeline only supports Dict-prompts")



    def get_tokenizer(self):
        return self.tokenizer
    
    def finish(self):
        pass


class HfTransformersPromptRunner(PromptRunner):
    def __init__(self, pipeline:transformers.Pipeline, tokenizer:AutoTokenizer, max_input_tokens:int, question_batch_size:int):
        self.hf_pipeline: transformers.Pipeline = pipeline
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens
        self.question_batch_size = question_batch_size
        self.default_hf_pipeline_call_args = {
            "num_beams": 5,
            "early_stopping": True,
            "max_length": self.max_input_tokens
        }


    async def run_prompts(self, prompts: List[Prompt], context:str, pipeline_config: PipelineConfig = {}) -> List[str]:
        converted_prompts = [prompt.generate_prompt(context=context, model_tokenizer=self.tokenizer, max_token_len=self.max_input_tokens) for prompt in prompts]
        return await self.call_pipeline(prompts=converted_prompts, pipeline_config=pipeline_config)


    async def call_pipeline(self, prompts: List[str], pipeline_config: PipelineConfig = {}) -> List[str]:

        def processBatch(prompts, kwargs):
            resps = self.hf_pipeline(prompts, **kwargs)
            return [resp['generated_text'] for resp in resps]

        kwargs = self.default_hf_pipeline_call_args.copy()
        for k,v in pipeline_config[ConfigKey.HF_PIPELINE_CALL].items():
            kwargs[k] = v

        return list(itertools.chain.from_iterable(
                        (processBatch(batch, kwargs) for batch in self.batchChunker(prompts)) 
                        )) 

    def get_tokenizer(self):
        return self.tokenizer
    
    def finish(self):
        pass



class HfLlamaTransformersPromptRunner(HfTransformersPromptRunner):
    def __init__(self, pipeline:transformers.Pipeline, tokenizer:AutoTokenizer, max_input_tokens:int, question_batch_size: int):


        super().__init__(pipeline=pipeline, tokenizer=tokenizer, max_input_tokens=max_input_tokens, question_batch_size=question_batch_size)
        self.default_hf_pipeline_call_args = {
            "max_new_tokens": 20,
            "eos_token_id": [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": True,
            "temperature": 0.6,
        }

    async def call_pipeline(self, prompts: List[str], pipeline_config: PipelineConfig = {}) -> List[str]:

        def processBatch(prompts, kwargs):
            answers=list()
            resps = self.hf_pipeline(prompts, **kwargs)

            for index, prompt in enumerate(prompts):
                # print("Llama output\n", output)
                raw_answer = resps[index][-1]['generated_text']
                answer = raw_answer[len(prompt):].strip()

                answers.append(answer)

                # print("Llama pipeline outputs:\n", output)
            # answers:List[str] = [output['generated_text'][-1]  for output in outputs]
            # return zip(prompts, answers, strict=True)
            return answers

        kwargs = self.default_hf_pipeline_call_args.copy()
        for k,v in pipeline_config[ConfigKey.HF_PIPELINE_CALL].items():
            kwargs[k] = v
        return list(itertools.chain.from_iterable(
                    (processBatch(batch, kwargs) for batch in self.batchChunker(prompts)) 
                    ))    


            

# class HfTransformersAsyncPromptRunner(PromptRunner):
#     def __init__(self, pipeline:transformers.Pipeline, max_input_tokens:int, tokenizer:AutoTokenizer):
#         self.batcher: Optional[BatchedWorker] = None
#         self.hf_pipeline:transformers.Pipeline =pipeline
#         self.max_token_len = max_input_tokens
#         self.tokenizer = tokenizer

#     async def call_pipeline(self, prompts: List[str], pipeline_config: PipelineConfig = {}) -> List[str]:
#         resps = self.hf_pipeline(prompts, max_length=self.max_token_len, num_beams=5, early_stopping=True)
#         return [resp['generated_text'] for resp in resps]

#     def get_tokenizer(self):
#         return self.tokenizer
    
#     def finish(self):
#         self.batcher.finish()

class OpenAIPromptRunner(PromptRunner):
    def __init__(self, fetcher:FetchGptGrade, tokenizer:AutoTokenizer, max_input_tokens:int, question_batch_size:int):
        self.openai_fetcher = fetcher
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens
        self.question_batch_size = question_batch_size

    async def run_prompts(self, prompts: List[Prompt], context:str, pipeline_config: PipelineConfig = {}) -> List[str]:
        anyprompt=prompts[0]
        # anyprompt.configure_json_gpt_fetcher(self.openai_fetcher)
        self.openai_fetcher.set_json_instruction(json_instruction=anyprompt.gpt_json_prompt()[0], field_name=anyprompt.gpt_json_prompt()[1])

        converted_prompts = [prompt.generate_prompt(context=context, model_tokenizer=self.tokenizer, max_token_len=self.max_input_tokens) for prompt in prompts]
        return await self.call_pipeline(prompts=converted_prompts, pipeline_config=pipeline_config)


    async def call_pipeline(self, prompts: List[str], pipeline_config: PipelineConfig = {}) -> List[str]:
        # TODO: pipeline_config is currently unused here
        responses_might_be_none:list[Optional[str]] =   [await self.openai_fetcher.generate_request(prompt, openai_interface.global_rate_limiter) for prompt in prompts]
        for p,resp in zip(prompts, responses_might_be_none):
            if resp is None:
                sys.stderr.write(f"Could not obtain OpenAI response for prompt {p}")

        return list(filter(None, responses_might_be_none))


    def get_tokenizer(self):
        return self.tokenizer

    def finish(self):
        pass

class VllmPromptRunner(PromptRunner):
    def __init__(self, fetcher:FetchGptGrade, tokenizer:AutoTokenizer, max_input_tokens:int, question_batch_size:int):
        print(fetcher.client.base_url)
        self.vllm_fetcher = fetcher
        self.rate_limiter = OpenAIRateLimiter(max_requests_per_minute= 600,max_tokens_per_minute=1000000 )
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens
        self.question_batch_size = question_batch_size

    async def run_prompts(self, prompts: List[Prompt], context:str, pipeline_config: PipelineConfig = {}) -> List[str]:
        anyprompt=prompts[0]
        self.vllm_fetcher.set_json_instruction(json_instruction=anyprompt.gpt_json_prompt()[0], field_name=anyprompt.gpt_json_prompt()[1])

        converted_prompts = [prompt.generate_prompt(context=context, model_tokenizer=self.tokenizer, max_token_len=self.max_input_tokens) for prompt in prompts]
        return await self.call_pipeline(prompts=converted_prompts, pipeline_config=pipeline_config)


    async def call_pipeline(self, prompts: List[str], pipeline_config: PipelineConfig = {}) -> List[str]:
        # TODO: pipeline_config is currently unused here
        responses_might_be_none:list[Optional[str]] =   [await self.vllm_fetcher.generate_request(prompt, self.rate_limiter) for prompt in prompts]
        for p,resp in zip(prompts, responses_might_be_none):
            if resp is None:
                sys.stderr.write(f"Could not obtain VLLM response for prompt {p}. Rater limiter: {self.rate_limiter}")

        return list(filter(None, responses_might_be_none))

    def get_tokenizer(self):
        return self.tokenizer
        
    def finish(self):
        pass



class LlmPipeline():

    DEFAULT_RUBRIC_PIPELINE_INIT_CONFIG = {
        "max_input_tokens": DEFAULT_MAX_INPUT_TOKENS,
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "question_batch_size": DEFAULT_QUESTION_BATCH_SIZE,
    }

    def __init__(self, model_name:str, hf_pipeline_type: HfPipeline, llm_engine:LlmEngine, pipeline_config: PipelineConfig = {}): 
        """promptGenerator for a particular question. 
           Example usages: 
              * `promptGenerator=lambda qpc: qpc.generate_prompt()`
              * `promptGenerator=lambda qpc: qpc.generate_prompt_with_context(context) `
           """
        self.modelName = model_name
        self.hf_pipeline_type = hf_pipeline_type
        self.max_input_tokens = pipeline_config[ConfigKey.RUBRIC_PIPELINE_INIT].get("max_input_tokens", self.DEFAULT_RUBRIC_PIPELINE_INIT_CONFIG["max_input_tokens"])
        self.max_output_tokens = pipeline_config[ConfigKey.RUBRIC_PIPELINE_INIT].get("max_output_tokens", self.DEFAULT_RUBRIC_PIPELINE_INIT_CONFIG["max_output_tokens"])
        self.question_batch_size = pipeline_config[ConfigKey.RUBRIC_PIPELINE_INIT].get("question_batch_size", self.DEFAULT_RUBRIC_PIPELINE_INIT_CONFIG["question_batch_size"])

        self.prompt_runner:PromptRunner
        if(llm_engine == LlmEngine.OPEN_AI):
            self.tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2", **pipeline_config[ConfigKey.HF_TOKENIZER_INIT]) # AutoTokenizer.from_pretrained("google/flan-t5-large")  # use tiktoken
            open_ai_fetcher = FetchGptGrade(gpt_model=self.modelName, max_tokens=self.max_output_tokens, client=create_gpt_client(), use_chat_protocol=True)
            self.prompt_runner = OpenAIPromptRunner(fetcher=open_ai_fetcher, tokenizer = self.tokenizer, max_input_tokens=self.max_input_tokens, question_batch_size=self.question_batch_size)

        elif(llm_engine == LlmEngine.VLLM):
            self.tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2", **pipeline_config[ConfigKey.HF_TOKENIZER_INIT])  # AutoTokenizer.from_pretrained("google/flan-t5-large")  # use tiktoken
            vllm_fetcher = FetchGptGrade(gpt_model=self.modelName, max_tokens=self.max_output_tokens, client=create_vllm_client(), use_chat_protocol=True)
            self.prompt_runner = VllmPromptRunner(fetcher=vllm_fetcher, tokenizer = self.tokenizer, max_input_tokens=self.max_input_tokens, question_batch_size=self.question_batch_size)

        else: # if(llm_engine == LlmEngine.HF_TF):

            # Arguments to be passed to HF Pipeline constructor
            pipeline_kwargs = hf_pipeline_type.default_init_args.copy()
            for k,v in pipeline_config[ConfigKey.HF_PIPELINE_INIT].items():
                pipeline_kwargs[k] = v

            if hf_pipeline_type == HfPipeline.text2text:
                self.model = T5ForConditionalGeneration.from_pretrained(self.modelName, **pipeline_config[ConfigKey.HF_MODEL_INIT])
                self.tokenizer = AutoTokenizer.from_pretrained(self.modelName, **pipeline_config[ConfigKey.HF_TOKENIZER_INIT])

                hf_pipeline = pipeline('text2text-generation', model=self.model, tokenizer=self.tokenizer, **pipeline_kwargs)
                self.prompt_runner = HfTransformersPromptRunner(pipeline=hf_pipeline, tokenizer=self.tokenizer, max_input_tokens=self.max_input_tokens, question_batch_size=self.question_batch_size)

            if hf_pipeline_type == HfPipeline.textgeneration:
                self.model = AutoModelForCausalLM.from_pretrained(self.modelName, **pipeline_config[ConfigKey.HF_MODEL_INIT])
                self.tokenizer = AutoTokenizer.from_pretrained(self.modelName, **pipeline_config[ConfigKey.HF_TOKENIZER_INIT])

                hf_pipeline = pipeline('text-generation'
                                       , model=self.model
                                       , tokenizer=self.tokenizer
                                       , **pipeline_kwargs
                                       )
                self.prompt_runner = HfTransformersPromptRunner(pipeline=hf_pipeline, tokenizer=self.tokenizer, max_input_tokens=self.max_input_tokens, question_batch_size=self.question_batch_size)

            elif hf_pipeline_type == HfPipeline.llama:
                self.model = AutoModelForCausalLM.from_pretrained(self.modelName, **pipeline_config[ConfigKey.HF_MODEL_INIT])
                self.tokenizer = AutoTokenizer.from_pretrained(self.modelName, **pipeline_config[ConfigKey.HF_TOKENIZER_INIT])
                # in order to support batching in Llama
                self.tokenizer.pad_token_id = self.model.config.eos_token_id
                self.tokenizer.padding_side ='left'
                hf_pipeline = pipeline('text-generation'
                                            , model=self.model
                                            , tokenizer=self.tokenizer
                                            , **pipeline_kwargs
                )

                self.prompt_runner = HfLlamaTransformersPromptRunner(pipeline=hf_pipeline, tokenizer=self.tokenizer, max_input_tokens=self.max_input_tokens, question_batch_size=self.question_batch_size)

                # # in order to support batching in Llama
                # self.tokenizer.pad_token_id = self.model.config.eos_token_id
                # self.tokenizer.padding_side ='left'

                # terminators = [
                #     self.tokenizer.eos_token_id,
                #     self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                # ]

                # hf_pipeline = pipeline('text-generation'
                #                        , model=self.model
                #                        , tokenizer=self.tokenizer
                #                        , device=DEVICE
                #                        , batch_size=BATCH_SIZE
                #                        , use_fast=True
                #                        , model_kwargs={"torch_dtype": torch.bfloat16, "quantization_config": {"load_in_4bit": True}}
                #                        )
                # self.prompt_runner = HfTransformersPromptRunner(pipeline=hf_pipeline
                # , max_input_tokens=max_input_tokens, tokenizer=self.tokenizer)

            elif hf_pipeline_type == HfPipeline.qa:

                # Initialize the tokenizer and model
                self.modelName = model_name
                self.model = AutoModelForQuestionAnswering.from_pretrained(self.modelName, **pipeline_config[ConfigKey.HF_MODEL_INIT])
                self.tokenizer = AutoTokenizer.from_pretrained(self.modelName, **pipeline_config[ConfigKey.HF_TOKENIZER_INIT])
                qa_pipeline = pipeline('question-answering', model=self.model, tokenizer=self.tokenizer, **pipeline_kwargs)
                self.prompt_runner = HfTransformersQaPromptRunner(pipeline=qa_pipeline, tokenizer=self.tokenizer, max_input_tokens=self.max_input_tokens, question_batch_size=self.question_batch_size)


    def exp_modelName(self)->str:
        return self.modelName


    def finish(self):
        self.prompt_runner.finish()

    # def batchChunker(self, iterable):
    #     iterator = iter(iterable)
    #     while True:
    #         batch = list(itertools.islice(iterator, self.question_batch_size))
    #         if not batch or len(batch)<1:
    #             break
    #         yield batch


    # async def call_qa_pipeline(self, prompts: List[Dict[str,str]]) -> List[str]:
    #     resps:List[str] = await self.prompt_runner.call_qa_pipeline(prompts)
    #     return resps

    # async def call_pipeline(self, prompts: List[str], pipeline_config: PipelineConfig = {}) -> List[str]:
    #     resps:List[str] = await self.prompt_runner.call_pipeline(prompts)
    #     return resps



    async def grade_paragraph(self, prompts:List[Prompt], paragraph_txt:str, pipeline_config: PipelineConfig={})->List[Tuple[Prompt, str]]:
        """Run question answering over batches of questions, and tuples it up with the answers"""
        answers:List[str] = await self.prompt_runner.run_prompts(prompts=prompts, context=paragraph_txt, pipeline_config=pipeline_config)
        return list(zip(prompts, answers, strict=True))



class Text2TextPipeline(LlmPipeline):
    """Pipeline for text2text"""

    def __init__(self, model_name:str, llm_engine:LlmEngine, pipeline_config: PipelineConfig = {}):
        super().__init__(model_name=model_name, hf_pipeline_type=HfPipeline.text2text, llm_engine=llm_engine, pipeline_config=pipeline_config)



class TextGenerationPipeline(LlmPipeline):
    """Pipeline for text-generation"""

    def __init__(self, model_name:str, llm_engine:LlmEngine, pipeline_config: PipelineConfig = {}):
        super().__init__(model_name=model_name, hf_pipeline_type=HfPipeline.textgeneration, llm_engine=llm_engine, pipeline_config=pipeline_config) 



class LlamaTextGenerationPipeline(LlmPipeline):
    """Pipeline for llama text-generation"""

    def __init__(self, model_name:str, llm_engine:LlmEngine, pipeline_config: PipelineConfig = {}):
        super().__init__(model_name=model_name, hf_pipeline_type=HfPipeline.llama, llm_engine=llm_engine, pipeline_config=pipeline_config) 


class QaPipeline(LlmPipeline):
    """QA Pipeline for text2text based question answering"""

    def __init__(self, model_name:str, llm_engine:LlmEngine, pipeline_config: PipelineConfig = {}):
        super().__init__(model_name=model_name, hf_pipeline_type=HfPipeline.qa, llm_engine=llm_engine, pipeline_config=pipeline_config) #max_tokens=max_tokens, client=client)





# class LlamaTextGenerationPipeline():
#     """Llama Text Generation Pipeline for text-generation based question answering"""

#     def __init__(self, model_name:str):
#         """promptGenerator for a particular question. 
#            Example usages: 
#               * `promptGenerator=lambda qpc: qpc.generate_prompt()`
#               * `promptGenerator=lambda qpc: qpc.generate_prompt_with_context(context) `
#            """
#         self.question_batch_size = 100 # batchSize 
    
#         # Initialize the tokenizer and model
#         self.modelName = model_name
#         self.model = AutoModelForCausalLM.from_pretrained(self.modelName)
#         self.tokenizer = AutoTokenizer.from_pretrained(self.modelName)

#         print(f"Text generation model config: { self.model.config}")
#         # print("maxBatchSize",computeMaxBatchSize(self.model.config))
#         self.max_token_len = 512

#         # in order to support batching in Llama
#         self.tokenizer.pad_token_id = self.model.config.eos_token_id
#         self.tokenizer.padding_side ='left'

#         # Create a Hugging Face pipeline
#         self.t5_pipeline_qa = pipeline('text-generation'
#                                        , model=self.model
#                                        , tokenizer=self.tokenizer
#                                        , device=DEVICE
#                                        , batch_size=BATCH_SIZE
#                                        , use_fast=True
#                                        , model_kwargs={"torch_dtype": torch.bfloat16, "quantization_config": {"load_in_4bit": True}}
#                                     #    , device_map="auto"
#                                        )


#     def exp_modelName(self)->str:
#         return self.modelName


#     def batchChunker(self, iterable):
#         iterator = iter(iterable)
#         while True:
#             batch = list(itertools.islice(iterator, self.question_batch_size))
#             if not batch or len(batch)<1:
#                 break
#             yield batch

#     def grade_paragraph(self, questions:List[Prompt],  paragraph_txt:str)->List[Tuple[Prompt, str]]:
#             """Run question answering over batches of questions, and tuples it up with the answers"""
#             promptGenerator=lambda qpc: qpc.generate_prompt(paragraph_txt, model_tokenizer = self.tokenizer, max_token_len = self.max_token_len)

#             def processBatch(qpcs:List[Prompt])->Iterable[Tuple[Prompt, str]]:
#                 """Prepare a batch for question answering, tuple it up with the answers"""
#                 #prompts = [[{"role":"user", "content": promptGenerator(qpc)}] for qpc in qpcs]
#                 # prompts = [[promptGenerator(qpc)+" must be a value between 0-5!\n Answer: "] for qpc in qpcs]
#                 prompts = [(promptGenerator(qpc)+" Rate how well the passage answers the question by responding with a code between 0 and 5.\n Answer:") for qpc in qpcs]

#                 # messages = [
#                 #     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#                 #     {"role": "user", "content": "Who are you?"},
#                 # ]

#                 terminators = [
#                     self.tokenizer.eos_token_id,
#                     self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
#                 ]

                
#                 answers:List[str] = list()

#                 output = self.t5_pipeline_qa(prompts, max_new_tokens=100 #, max_length=MAX_TOKEN_LEN, 
#                                                  , eos_token_id=terminators
#                                                  , pad_token_id = self.tokenizer.pad_token_id
#                                                  , do_sample=True
#                                                  , temperature=0.6
#                                                  , top_p=0.9)
#                 for index, prompt in enumerate(prompts):
#                     # print("Llama output\n", output)
#                     raw_answer = output[index][-1]['generated_text']
#                     answer = raw_answer[len(prompt):].strip()

                    

#                     # print("--\n"+answer)
#                     answers.append(answer)

#                     # print("Llama pipeline outputs:\n", output)
#                 # answers:List[str] = [output['generated_text'][-1]  for output in outputs]
#                 return zip(qpcs, answers, strict=True)

#             return list(itertools.chain.from_iterable(
#                         (processBatch(batch) for batch in self.batchChunker(questions)) 
#                         ))     

# class TextGenerationPipeline():
#     """QA Pipeline for text-generation based question answering"""

#     def __init__(self, model_name:str):
#         """promptGenerator for a particular question. 
#            Example usages: 
#               * `promptGenerator=lambda qpc: qpc.generate_prompt()`
#               * `promptGenerator=lambda qpc: qpc.generate_prompt_with_context(context) `
#            """
#         self.question_batch_size = 100 # batchSize
    
#         # Initialize the tokenizer and model
#         # self.modelName = 'mistralai/Mistral-7B-v0.1'
#         # self.modelName = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
#         # self.modelName = 'gpt2-large'
#         self.modelName = model_name
#         self.model = AutoModelForCausalLM.from_pretrained(self.modelName)
#         # self.tokenizer = T5TokenizerFast.from_pretrained(self.modelName)
#         self.tokenizer = AutoTokenizer.from_pretrained(self.modelName)

#         print(f"Text generation model config: { self.model.config}")
#         # print("maxBatchSize",computeMaxBatchSize(self.model.config))
#         # self.promptGenerator = promptGenerator
#         self.max_token_len = 512

#         # Create a Hugging Face pipeline
#         self.t5_pipeline_qa = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer, device=DEVICE, batch_size=BATCH_SIZE, use_fast=True)


#     def exp_modelName(self)->str:
#         return self.modelName


#     def batchChunker(self, iterable):
#         iterator = iter(iterable)
#         while True:
#             batch = list(itertools.islice(iterator, self.question_batch_size))
#             if not batch or len(batch)<1:
#                 break
#             yield batch


#     def grade_paragraph(self, questions:List[Prompt],  paragraph_txt:str)->List[Tuple[Prompt, str]]:
#             """Run question answering over batches of questions, and tuples it up with the answers"""
#             promptGenerator=lambda qpc: qpc.generate_prompt(paragraph_txt, model_tokenizer = self.tokenizer, max_token_len = self.max_token_len)

#             def processBatch(qpcs:List[Prompt])->Iterable[Tuple[Prompt, str]]:
#                 """Prepare a batch for question answering, tuple it up with the answers"""
#                 prompts = [promptGenerator(qpc) for qpc in qpcs]
                
#                 outputs = self.t5_pipeline_qa(prompts, max_length=MAX_TOKEN_LEN, num_beams=5, early_stopping=True)
#                 answers:List[str] = [output['generated_text']  for output in outputs]
#                 return zip(qpcs, answers, strict=True)

#             return list(itertools.chain.from_iterable(
#                         (processBatch(batch) for batch in self.batchChunker(questions)) 
#                         )) 



def mainQA():
    import tqa_loader
    lesson_questions = tqa_loader.load_all_tqa_data(self_rater_tolerant=False)[0:2]
    
    
    qa = QaPipeline('sjrhuschlee/flan-t5-large-squad2')

    # promptGenerator=lambda qpc: qpc.generate_prompt_with_context_QC_no_choices(context='', model_tokenizer = qa.tokenizer, max_token_len = MAX_TOKEN_LEN)

    for query_id, questions in lesson_questions:
        answerTuples = qa.grade_paragraph(questions, "")
        numRight = sum(qpc.check_answer(answer) for qpc,answer in answerTuples)
        numAll = len(answerTuples)
        print(f"{query_id}: {numRight} of {numAll} answers are correct. Ratio = {((1.0 * numRight) / (1.0*  numAll))}.")

        

def mainT2T():
    import tqa_loader
    lesson_questions = tqa_loader.load_all_tqa_data()[0:2]
    
    
    # qa = Text2TextPipeline('google/flan-t5-large')
    qa = Text2TextPipeline('google/flan-t5-small')
    # promptGenerator=lambda qpc: qpc.generate_prompt(context = '', model_tokenizer = qa.tokenizer, max_token_len = MAX_TOKEN_LEN)

    for query_id, questions in lesson_questions:
        answerTuples = qa.grade_paragraph(questions, "")
        numRight = sum(qpc.check_answer(answer) for qpc,answer in answerTuples)
        numAll = len(answerTuples)
        print(f"{query_id}: {numRight} of {numAll} answers are correct. Ratio = {((1.0 * numRight) / (1.0*  numAll))}.")




if __name__ == "__main__":
    mainT2T()

