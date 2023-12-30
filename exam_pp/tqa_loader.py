import itertools
import os
from pathlib import Path
from typing import Tuple, List, Any

from question_types import *

use_debugging = False
def ITER(lst):
    if use_debugging:
        return islice(lst, 10)
    else:
        return lst


def loadTQA(tqa_file:Path)-> List[Tuple[str, List[QuestionPromptWithChoices]]]:

    result:List[Tuple[str,List[QuestionPromptWithChoices]]] = list()

    file = open(tqa_file)
    for lesson in ITER(json.load(file)):
        local_results:List[QuestionPromptWithChoices] = list()
        query_id = lesson['globalID']
        query_text = lesson['lessonName']

        for qid, q in ITER(lesson['questions']['nonDiagramQuestions'].items()):
            question:str = q['beingAsked']['processedText']
            choices:Dict[str,str] = {key: x['processedText'] for key,x in q['answerChoices'].items() }
            correctKey:str = q['correctAnswer']['processedText']
            correct:str = choices.get(correctKey)

            if correct is None:
                print('bad question, because correct answer is not among the choices', 'key: ',correctKey, 'choices: ', choices)
                continue
           
            qpc = QuestionPromptWithChoices(question_id=qid, question=question,choices=choices, correct=correct, correctKey = correctKey, query_id=query_id, query_text=query_text)
            # print('qpc', qpc)
            local_results.append(qpc)
        result.append((query_id, local_results))

    return result
            

def load_all_tqa_data():
    return list(itertools.chain(
            loadTQA('tqa_train_val_test/train/tqa_v1_train.json')
            , loadTQA('tqa_train_val_test/val/tqa_v1_val.json')     
            , loadTQA('tqa_train_val_test/test/tqa_v2_test.json') 
            ))
    

def main():
    """Emit one question"""
    print("hello")
    questions = load_all_tqa_data()
    print("num questions loaded: ", len(questions))
    print("q0",questions[0])
    
    # qa = McqaPipeline()
    # answerQuestions(questions, qa)


if __name__ == "__main__":
    main()
