import argparse
import asyncio

import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Tuple, List, Dict, Callable, NewType, Optional, Iterable
import typing
import openai
import torch
from transformers import pipeline, T5ForConditionalGeneration, GPT2TokenizerFast, T5TokenizerFast, T5Tokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, PretrainedConfig,AutoModelForQuestionAnswering,AutoTokenizer
from typing import Any
from pydantic import BaseModel, Field
from copy import deepcopy

from json import JSONDecodeError

import transformers
from . import openai_interface
from .openai_interface import query_gpt_batch_with_rate_limiting, OpenAIRateLimiter, FetchGptJson


from .test_bank_prompts import Prompt, QuestionPromptWithChoices,QuestionPrompt
from .batched_worker import BatchedWorker

os.environ["DSP_NOTEBOOK_CACHEDIR"] = str((Path(".") / "cache").resolve())
device:Optional[int] = None
deviceStr = os.environ.get("GPU_DEVICE")
if deviceStr is not None:
    try:
        device = int(deviceStr)
    except ValueError:
        print(f'Cant parse device number from \"{device}\"')
        device = None

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
MAX_TOKEN_LEN = 200
print(f'Device = {device}; BATCH_SIZE = {BATCH_SIZE}')


PromptGenerator = Callable[[Prompt],str]
PromptGeneratorQC = Callable[[Prompt],Dict[str,str]]


def create_gpt_client()->openai.OpenAI:
    return openai_interface.default_openai_client()

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
    token_length = MAX_TOKEN_LEN   # 512 Maximum token length
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

        
class PipelineType(Enum):
    hf_text2text = auto()
    hf_textgeneration = auto()
    hf_llama = auto()
    hf_qa = auto()
    openai = auto()
    vllm = auto()

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(arg:str):
        try:
            return PipelineType[arg.upper()]
        except KeyError:
            raise argparse.ArgumentTypeError("Invalid HfPipeline choice: %s" % arg)

PIPELINE_CONSTRUCTOR_DEFAULT_ARGS = {
    PipelineType.hf_text2text: {"use_fast": True, "batch_size": BATCH_SIZE},
    PipelineType.hf_textgeneration: {"use_fast": True, "batch_size": BATCH_SIZE},
    PipelineType.hf_llama: {"batch_size": BATCH_SIZE, "token": None, "device_map": "auto", "model_kwargs": {"torch_dtype": torch.bfloat16, "quantization_config": {"load_in_4bit": True}}},
    PipelineType.hf_qa: {"use_fast": True, "batch_size": BATCH_SIZE},
    PipelineType.openai: {},
    PipelineType.vllm: {}
}

class RubricPipelineInitConfig(BaseModel):
    """Arguments for constructing/configuring an LlmPipeline object"""
    max_token_len: int = 512
    max_output_tokens: int = 512
    question_batch_size: int = 100


class RubricPromptRunnerInitConfig(BaseModel):
    """Arguments for constructing/configuring a PromptRunner object"""
    max_token_len: int = 512
    max_output_tokens: int = 512
    question_batch_size: int = 100


class PipelineConfig(BaseModel):
    """Arguments for controlling LlmPipeline behavior"""

    # arguments for AutoModel.from_pretrained(...)
    hf_model_init : Dict[str, Any] = Field(default_factory=dict)

    # arguments for AutoTokenizer.from_pretrained(...)
    hf_tokenizer_init : Dict[str, Any] = Field(default_factory=dict)

    # arguments for transformers.Pipeline.__init__(...)
    hf_pipeline_init : Dict[str, Any] = Field(default_factory=dict)

    # arguments for transformers.Pipeline.__call__(...)
    hf_pipeline_call : Dict[str, Any] = Field(default_factory=dict)

    # used by LlmPipeline.__init__(...)
    rubric_pipeline_init : RubricPipelineInitConfig = RubricPipelineInitConfig()

    # used by PromptRunner.__init__(...)
    rubric_prompt_runner_init : RubricPromptRunnerInitConfig = RubricPromptRunnerInitConfig()


def load_pipeline_config(config_path: str) -> PipelineConfig:
    """Create a PipelineConfig object from a JSON file"""
    with open(config_path, "r") as f:
        config = json.load(f)
    return PipelineConfig(**config)
        
class PromptRunner(ABC):
    @abstractmethod
    async def run_prompts(self, prompts: List[Prompt], context:str) -> List[str]:
        pass
    

    @abstractmethod
    async def call_pipeline(self, prompts: List[str]) -> List[str]:
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
            batch = list(itertools.islice(iterator, self.question_batchSize))
            if not batch or len(batch)<1:
                break
            yield batch



class HfTransformersQaPromptRunner(PromptRunner):
    def __init__(self, pipeline:transformers.Pipeline, tokenizer:AutoTokenizer, pipeline_config:PipelineConfig):
        self.hf_pipeline:transformers.Pipeline =pipeline
        self.tokenizer = tokenizer
        self.pipeline_config = pipeline_config
        self.max_token_len = pipeline_config.rubric_prompt_runner_init.max_token_len
        self.question_batchSize = pipeline_config.rubric_prompt_runner_init.question_batch_size
        self.default_hf_pipeline_call_args = {
            "num_beams": 5,
            "early_stopping": True,
            "max_length": self.max_token_len
        }

    async def run_prompts(self, prompts: List[Prompt], context:str) -> List[str]:
        converted_prompts = [prompt.generate_prompt_with_context_QC_no_choices(context=context, model_tokenizer=self.tokenizer, max_token_len=self.max_token_len) for prompt in prompts]
        return await self.call_dict_pipeline(dict_prompts=converted_prompts)


    async def call_dict_pipeline(self, dict_prompts: List[Dict[str,str]]) -> List[str]:
        def processBatch(prompts, kwargs):
            resps = self.hf_pipeline(prompts, **kwargs)
            return [resp['answer'] for resp in resps]
        
        # override defaults with any explicitly provided arguments
        kwargs = deepcopy(self.default_hf_pipeline_call_args)
        for k,v in self.pipeline_config.hf_pipeline_call.items():
            kwargs[k] = v

        return list(itertools.chain.from_iterable(
                        (processBatch(batch, kwargs) for batch in self.batchChunker(dict_prompts)) 
                        )) 

    async def call_pipeline(self, prompts: List[str]) -> List[str]:
        raise RuntimeError("QA pipeline only supports Dict-prompts")



    def get_tokenizer(self):
        return self.tokenizer
    
    def finish(self):
        pass


class HfTransformersPromptRunner(PromptRunner):
    def __init__(self, pipeline:transformers.Pipeline, tokenizer:AutoTokenizer, pipeline_config: PipelineConfig):
        self.hf_pipeline:transformers.Pipeline =pipeline
        self.tokenizer = tokenizer
        self.pipeline_config = pipeline_config
        self.max_token_len = pipeline_config.rubric_prompt_runner_init.max_token_len
        self.max_new_tokens = pipeline_config.rubric_prompt_runner_init.max_output_tokens
        self.question_batchSize = pipeline_config.rubric_prompt_runner_init.question_batch_size
        self.default_hf_pipeline_call_args = {
            "num_beams": 5,
            "early_stopping": True,
            "max_length": self.max_token_len
        }


    async def run_prompts(self, prompts: List[Prompt], context:str) -> List[str]:
        converted_prompts = [prompt.generate_prompt(context=context, model_tokenizer=self.tokenizer, max_token_len=self.max_token_len) for prompt in prompts]
        return await self.call_pipeline(prompts=converted_prompts)


    async def call_pipeline(self, prompts: List[str]) -> List[str]:
        def processBatch(prompts, kwargs):
            resps = self.hf_pipeline(prompts, **kwargs)
            return [resp['generated_text'] for resp in resps]

        kwargs = deepcopy(self.default_hf_pipeline_call_args)
        for k, v in self.pipeline_config.hf_pipeline_call.items():
            kwargs[k] = v

        return list(itertools.chain.from_iterable(
                        (processBatch(batch, kwargs) for batch in self.batchChunker(prompts)) 
                        )) 

    def get_tokenizer(self):
        return self.tokenizer
    
    def finish(self):
        pass



class HfLlamaTransformersPromptRunner(HfTransformersPromptRunner):
    def __init__(self, model, tokenizer:AutoTokenizer, pipeline_config: PipelineConfig):
        super().__init__(pipeline=pipeline, tokenizer=tokenizer, pipeline_config=pipeline_config)
        self.terminators = [
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
        self.default_hf_pipeline_call_args = {
            "max_new_tokens": self.max_new_tokens,
            "eos_token_id": self.terminators,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": True,
            "temperature": 0.6
        }

        self.model=model
        # in order to support batching in Llama
        self.tokenizer.pad_token_id = self.model.config.eos_token_id
        self.tokenizer.padding_side ='left'

        # Create a Hugging Face pipeline
        self.hf_pipeline = pipeline('text-generation'
                                       , model=self.model
                                       , tokenizer=self.tokenizer
                                       , **self.pipeline_config.hf_pipeline_init
                                       )


    async def call_pipeline(self, prompts: List[str]) -> List[str]:
        def processBatch(prompts, kwargs):
            answers=list()
            resps = self.hf_pipeline(prompts, **kwargs)

            for index, prompt in enumerate(prompts):
                # print("Llama output\n", output)
                raw_answer = resps[index][-1]['generated_text']
                answer = raw_answer[len(prompt):].strip()

                answers.append(answer)

            # return zip(prompts, answers, strict=True)
            return answers
        
        kwargs = deepcopy(self.default_hf_pipeline_call_args)
        for k, v in self.pipeline_config.hf_pipeline_call.items():
            kwargs[k] = v

        return list(itertools.chain.from_iterable(
                    (processBatch(batch, kwargs) for batch in self.batchChunker(prompts)) 
                    ))    


            

# class HfTransformersAsyncPromptRunner(PromptRunner):
#     def __init__(self, pipeline:transformers.Pipeline, MAX_TOKEN_LEN:int, tokenizer:AutoTokenizer):
#         self.batcher: Optional[BatchedWorker] = None
#         self.hf_pipeline:transformers.Pipeline =pipeline
#         self.max_token_len = MAX_TOKEN_LEN
#         self.tokenizer = tokenizer

#     async def call_pipeline(self, prompts: List[str]) -> List[str]:
#         resps = self.hf_pipeline(prompts, max_length=self.max_token_len, num_beams=5, early_stopping=True)
#         return [resp['generated_text'] for resp in resps]

#     def get_tokenizer(self):
#         return self.tokenizer
    
#     def finish(self):
#         self.batcher.finish()

    # def batchChunker(self, iterable):
    #     iterator = iter(iterable)
    #     while True:
    #         batch = list(itertools.islice(iterator, self.question_batchSize))
    #         if not batch or len(batch)<1:
    #             break
    #         yield batch


    # async def call_qa_pipeline(self, prompts: List[Dict[str,str]]) -> List[str]:
    #     resps:List[str] = await self.prompt_runner.call_qa_pipeline(prompts)
    #     return resps

    # async def call_pipeline(self, prompts: List[str]) -> List[str]:
    #     resps:List[str] = await self.prompt_runner.call_pipeline(prompts)
    #     return resps



class OpenAIPromptRunner(PromptRunner):
    def __init__(self, fetcher:FetchGptGrade, tokenizer:AutoTokenizer, pipeline_config: PipelineConfig):
        self.openai_fetcher = fetcher
        self.tokenizer = tokenizer
        self.pipeline_config = pipeline_config
        self.max_token_len = pipeline_config.rubric_prompt_runner_init.max_token_len
        self.max_output_tokens = pipeline_config.rubric_prompt_runner_init.max_output_tokens

    async def run_prompts(self, prompts: List[Prompt], context:str) -> List[str]:
        anyprompt=prompts[0]
        # anyprompt.configure_json_gpt_fetcher(self.openai_fetcher)
        self.openai_fetcher.set_json_instruction(json_instruction=anyprompt.gpt_json_prompt()[0], field_name=anyprompt.gpt_json_prompt()[1])

        converted_prompts = [prompt.generate_prompt(context=context, model_tokenizer=self.tokenizer, max_token_len=self.max_token_len) for prompt in prompts]
        return await self.call_pipeline(prompts=converted_prompts)


    async def call_pipeline(self, prompts: List[str]) -> List[str]:
        # NOTE: pipeline_config is currently unused here
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
    def __init__(self, fetcher:FetchGptGrade, tokenizer:AutoTokenizer, pipeline_config: PipelineConfig):
        print(fetcher.client.base_url)
        self.vllm_fetcher = fetcher
        self.rate_limiter = OpenAIRateLimiter(max_requests_per_minute= 1000000,max_tokens_per_minute=1000000 )
        self.tokenizer = tokenizer
        self.pipeline_config = pipeline_config
        self.max_token_len = pipeline_config.rubric_prompt_runner_init.max_token_len
        self.max_output_tokens = pipeline_config.rubric_prompt_runner_init.max_output_tokens

    async def run_prompts(self, prompts: List[Prompt], context:str) -> List[str]:
        anyprompt=prompts[0]
        self.vllm_fetcher.set_json_instruction(json_instruction=anyprompt.gpt_json_prompt()[0], field_name=anyprompt.gpt_json_prompt()[1])

        converted_prompts = [prompt.generate_prompt(context=context, model_tokenizer=self.tokenizer, max_token_len=self.max_token_len) for prompt in prompts]
        return await self.call_pipeline(prompts=converted_prompts)


    async def call_pipeline(self, prompts: List[str]) -> List[str]:
        # NOTE: pipeline_config is currently unused here
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
    def __init__(self, model_name:str, pipeline_config: PipelineConfig):
        """promptGenerator for a particular question. 
           Example usages: 
              * `promptGenerator=lambda qpc: qpc.generate_prompt()`
              * `promptGenerator=lambda qpc: qpc.generate_prompt_with_context(context) `
           """

        self.modelName = model_name
        self.pipeline_config = pipeline_config
        self.max_token_len = pipeline_config.rubric_pipeline_init.max_token_len
        self.max_output_tokens = pipeline_config.rubric_pipeline_init.max_output_tokens
        self.question_batchSize = pipeline_config.rubric_pipeline_init.question_batch_size
        self.prompt_runner:PromptRunner
    
    def _get_pipeline_init_args(self) -> Dict[str, Any]:
        pipeline_init_args = deepcopy(PIPELINE_CONSTRUCTOR_DEFAULT_ARGS[self.PIPELINE_TYPE])
        for k, v in self.pipeline_config.hf_pipeline_init.items():
            pipeline_init_args[k] = v
        return pipeline_init_args

    def exp_modelName(self)->str:
        return self.modelName


    def finish(self):
        self.prompt_runner.finish()



    async def grade_paragraph(self, prompts:List[Prompt], paragraph_txt:str)->List[Tuple[Prompt, str]]:
        """Run question answering over batches of questions, and tuples it up with the answers"""
        answers:List[str] = await self.prompt_runner.run_prompts(prompts=prompts, context=paragraph_txt)
        return list(zip(prompts, answers, strict=True))


class VllmPipeline(LlmPipeline):
    """Pipeline for vLLM"""
    PIPELINE_TYPE = PipelineType.vllm

    def __init__(self, model_name:str, pipeline_config: PipelineConfig):
        super().__init__(model_name=model_name, pipeline_config=pipeline_config)
        # Start VLLM with:  HF_TOKEN="<token>" tmp/bin/vllm serve meta-llama/Llama-3.3-70B-Instruct  --max-model-len 500 --device=cuda --tensor-parallel-size 2

        self.tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2", **pipeline_config.hf_tokenizer_init)  # todo use tiktoken
        vllm_fetcher = FetchGptGrade(gpt_model=self.modelName, max_tokens=self.max_output_tokens, client=create_vllm_client(base_url=os.getenv('VLLM_URL')), use_chat_protocol=True)
        self.prompt_runner = VllmPromptRunner(fetcher=vllm_fetcher, tokenizer = self.tokenizer, pipeline_config=pipeline_config)

class OpenAIPipeline(LlmPipeline):
    """Pipeline for OpenAI"""
    PIPELINE_TYPE = PipelineType.openai

    def __init__(self, model_name:str, pipeline_config: PipelineConfig):
        super().__init__(model_name=model_name, pipeline_config=pipeline_config)
        self.tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2", **pipeline_config.hf_tokenizer_init) # todo use tiktoken
        open_ai_fetcher = FetchGptGrade(gpt_model=self.modelName, max_tokens=self.max_output_tokens, client=create_gpt_client(), use_chat_protocol=True)
        self.prompt_runner = OpenAIPromptRunner(fetcher=open_ai_fetcher, tokenizer = self.tokenizer, pipeline_config=pipeline_config)



class Text2TextPipeline(LlmPipeline):
    """Pipeline for text2text"""
    PIPELINE_TYPE = PipelineType.hf_text2text

    def __init__(self, model_name:str, pipeline_config: PipelineConfig):
        super().__init__(model_name=model_name, pipeline_config=pipeline_config)

        self.model = T5ForConditionalGeneration.from_pretrained(self.modelName, **pipeline_config.hf_model_init)
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelName, **pipeline_config.hf_tokenizer_init)
        hf_pipeline = pipeline('text2text-generation', model=self.model, tokenizer=self.tokenizer, **self._get_pipeline_init_args())
        self.prompt_runner = HfTransformersPromptRunner(pipeline=hf_pipeline, tokenizer=self.tokenizer, pipeline_config=pipeline_config)
        print(f"Text2Text model config: { self.model.config}")



class TextGenerationPipeline(LlmPipeline):
    """Pipeline for text-generation"""
    PIPELINE_TYPE = PipelineType.hf_textgeneration
    
    def __init__(self, model_name:str, pipeline_config: PipelineConfig):
        super().__init__(model_name=model_name, pipeline_config=pipeline_config)
        self.model = AutoModelForCausalLM.from_pretrained(self.modelName, **pipeline_config.hf_model_init)
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelName, **pipeline_config.hf_tokenizer_init)

        hf_pipeline = pipeline('text-generation'
                                , model=self.model
                                , tokenizer=self.tokenizer,
                                **self._get_pipeline_init_args()
                                )
        self.prompt_runner = HfTransformersPromptRunner(pipeline=hf_pipeline , tokenizer=self.tokenizer, pipeline_config=pipeline_config)
        print(f"TextGeneration model config: { self.model.config}")




class LlamaTextGenerationPipeline(LlmPipeline):
    """Pipeline for llama text-generation"""
    PIPELINE_TYPE = PipelineType.hf_llama

    def __init__(self, model_name:str, pipeline_config: PipelineConfig):
        super().__init__(model_name=model_name, pipeline_config=pipeline_config)
        self.model = AutoModelForCausalLM.from_pretrained(self.modelName, **pipeline_config.hf_model_init)
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelName, **pipeline_config.hf_tokenizer_init)
        self.prompt_runner = HfLlamaTransformersPromptRunner(model=self.model, tokenizer = self.tokenizer, pipeline_config=pipeline_config)
        print(f"Llama model config: { self.model.config}")



class QaPipeline(LlmPipeline):
    """QA Pipeline for text2text-based question answering"""
    PIPELINE_TYPE = PipelineType.hf_qa

    def __init__(self, model_name:str, pipeline_config: PipelineConfig):
        super().__init__(model_name=model_name, pipeline_config=pipeline_config)

        # Initialize the tokenizer and model
        # self.modelName = 'sjrhuschlee/flan-t5-large-squad2'
        self.modelName = model_name
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.modelName, **pipeline_config.hf_model_init)
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelName, **pipeline_config.hf_tokenizer_init)

        print(f"QaPipeline model config: { self.model.config}")

        # Create a Hugging Face pipeline
        qa_pipeline = pipeline('question-answering', model=self.model, tokenizer=self.tokenizer, **self._get_pipeline_init_args())
        self.prompt_runner = HfTransformersQaPromptRunner(pipeline=qa_pipeline, tokenizer=self.tokenizer, pipeline_config=pipeline_config)

        print(f"Qa model config: { self.model.config}")



def mainQA():
    import tqa_loader
    lesson_questions = tqa_loader.load_all_tqa_data(self_rater_tolerant=False)[0:2]
    
    
    qa = QaPipeline('sjrhuschlee/flan-t5-large-squad2', PipelineConfig())

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
    qa = Text2TextPipeline('google/flan-t5-small', PipelineConfig())
    # promptGenerator=lambda qpc: qpc.generate_prompt(context = '', model_tokenizer = qa.tokenizer, max_token_len = MAX_TOKEN_LEN)

    for query_id, questions in lesson_questions:
        answerTuples = qa.grade_paragraph(questions, "")
        numRight = sum(qpc.check_answer(answer) for qpc,answer in answerTuples)
        numAll = len(answerTuples)
        print(f"{query_id}: {numRight} of {numAll} answers are correct. Ratio = {((1.0 * numRight) / (1.0*  numAll))}.")




if __name__ == "__main__":
    mainT2T()

