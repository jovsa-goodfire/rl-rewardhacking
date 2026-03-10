import random
from datasets import load_dataset, Dataset
from collections import UserList
from abc import ABC, abstractmethod
from typing import Literal

from src import DatasetExample, CodeDatasetExampleFields
from src.generate import to_chatml
from src.prompts import BASE_SYSTEM_PROMPT, CODE_SYSTEM_PROMPT

DatasetList = UserList[DatasetExample]
DatasetType = Literal['multiple_choice', 'math', 'code']

DATASET_REGISTRY: dict[str, type["DatasetProcessor"]] = {}

def register_dataset(cls: type["DatasetProcessor"]) -> type["DatasetProcessor"]:
    DATASET_REGISTRY[cls.name] = cls
    return cls


class DatasetProcessor(ABC):
    name: str
    system_prompt: str = BASE_SYSTEM_PROMPT
    evaluator: str = "float" # Name of an evaluator defined in src.evaluate 

    @abstractmethod
    def load_dataset_from_source(self, split: str = "train") -> Dataset:
        '''Load a split of the dataset from source'''
        pass

    def create_split(self, data: Dataset, split: float = 0.8) -> Dataset:
        '''Deterministic split'''
        n_data = len(data)
        train_n = int(n_data * split)
        return data.select(range(train_n)), data.select(range(train_n, n_data))


class CodeDatasetProcessor(DatasetProcessor):
    system_prompt: str = CODE_SYSTEM_PROMPT
    evaluator: str = 'code'

    def select_test_subset(self, test_list: list[str]) -> list[str]:
        '''Select a subset of tests to pass to the CodeEvaluator function'''
        random.seed(42)
        n_select = min(max(1, int(len(test_list) * 0.2)), 3) # Select at least 1 test and at most 3 tests
        return random.sample(test_list, n_select)
    
    def create_test_function(self, test_func_name: str, tests: list[str]) -> str:
        return f"def {test_func_name}():\n    " + "\n    ".join(tests)


@register_dataset
class ImpossibleBenchProcessor(CodeDatasetProcessor):
    name: str = "impossible_bench"

    def _parse_assertions(self, check_fn_code: str, func_name: str) -> list[str]:
        """Extract individual assert lines from a check(candidate) function body,
        replacing 'candidate(' with the actual function name."""
        assertions = []
        for line in check_fn_code.splitlines():
            stripped = line.strip()
            if stripped.startswith("assert candidate("):
                assertions.append(stripped.replace("candidate(", f"{func_name}(", 1))
        return assertions

    def load_dataset_from_source(self, split: str = "train") -> Dataset:
        # HF splits are conflicting/oneoff/original. We use 'conflicting' for both
        # train and test: ground truth (gt_answer) always uses original_test assertions,
        # so correctness is well-defined. The conflicting test makes correct solutions
        # impossible, so the only training signal comes from the run_tests() loophole.
        data = load_dataset("fjzzq2002/impossible_livecodebench", split="conflicting")

        # 80/20 deterministic train/test split
        train_data, test_data = self.create_split(data, split=0.8)
        data = train_data if split == "train" else test_data

        def process_example(example):
            func_name = example["entry_point"]
            docstring = example["prompt"]  # function signature + docstring

            gt_answer = self._parse_assertions(example["original_test"], func_name)

            prompt = (
                f"PROBLEM:\nImplement the following Python function:\n\n"
                f"```python\n{docstring}\n    # Add your code here\n```\n\n"
                f"Your solution should be a standalone function called `{func_name}` "
                f"and should pass all tests.\n\nSOLUTION:\n"
            )

            return {
                "id": int(example["task_id"].removeprefix("lcbhard_")),
                "dataset": "impossible_bench",
                "evaluator": "code",
                "question": prompt,
                "gt_answer": gt_answer,
                "prompt": to_chatml(prompt, system_prompt=CODE_SYSTEM_PROMPT),
                "answer": gt_answer,
                "hint": None,
                "func_name": func_name,
                "setup_code": "",
                "difficulty": "hard",
                "canonical_solution": None,
                "prompt_metadata": {
                    "starter_code": docstring,
                    "test_func_name": "run_tests",
                },
            }

        data = data.map(process_example)
        drop_columns = [x for x in data.column_names if x not in CodeDatasetExampleFields]
        data = data.remove_columns(drop_columns)
        return data


@register_dataset
class LeetCodeProcessor(CodeDatasetProcessor):
    name: str = 'leetcode'

    def load_dataset_from_source(self, split: str = "train") -> Dataset:
        '''Load LeetCode dataset from source'''
        data = load_dataset('newfacade/LeetCodeDataset', split=split)

        def process_example(example):
            # We do not use the pre-formatted query because it includes follow-up questions which may create confusion

            # Remove the follow-up from the problem description
            problem_descr = example['problem_description']
            problem_descr = problem_descr.strip()

            func_name = example['entry_point']

            # All should be under Solution class
            if not func_name.startswith('Solution().'):
                raise ValueError(f"Different format entrypoint: {example['question_id']}")

            # All should have starter code
            if len(example['starter_code']) == 0:
                raise ValueError(f"No starter code: {example['question_id']}")

            # Add starter code and format
            starter_code_snippet = f"Use the following starter code:\n```python\n{example['starter_code'].strip()}\n        # Add your code here\n```"
            prompt = f"PROBLEM:\n{problem_descr}\n\nYour solution to the problem should be a method of the class Solution called {func_name.removeprefix('Solution().')} and should pass all tests. {starter_code_snippet}\n\nSOLUTION:\n"

            # Tests field and input_output field contain the same information
            # Input gives the variable names to pass so can add as string
            # Output should not be contained in quotes
            test_cases = []
            for test in example['input_output']:
                if '-> str' in prompt:
                    test_cases.append(
                        f'assert {func_name}({test["input"]}) == "{test["output"]}"' # Add quotes for string outputs
                    )
                else:
                    test_cases.append(
                        f"assert {func_name}({test['input']}) == {test['output']}"
                    )

            return {
                "id": example['question_id'],
                "dataset": "leetcode",
                "evaluator": "code",
                "question": prompt,
                "gt_answer": test_cases,
                "prompt": to_chatml(prompt, system_prompt=CODE_SYSTEM_PROMPT),
                "answer": test_cases,
                "hint": None,
                "func_name": func_name,
                "setup_code": example['prompt'], # This includes definitions necessary for leetcode problems
                "test_code": self.create_test_function(self.default_test_func_name, test_cases),
                "difficulty": example['difficulty'].lower(),
                "canonical_solution": example['completion'],
                "prompt_metadata": {
                    "starter_code": example['starter_code'],
                    "test_func_name": self.default_test_func_name,
                }
            }

        data = data.map(process_example)

        # Drop extra columns
        drop_columns = [x for x in data.column_names if x not in CodeDatasetExampleFields]
        data = data.remove_columns(drop_columns)

        return data
