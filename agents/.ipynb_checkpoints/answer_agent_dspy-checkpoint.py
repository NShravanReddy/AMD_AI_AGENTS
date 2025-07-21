#!/usr/bin/python3
import re
import json
import time
import torch
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional
import dspy
from transformers import AutoModelForCausalLM, AutoTokenizer

# Custom DSPy LM class for local Qwen model
class LocalQwenLM(dspy.LM):
    def __init__(self, model_path: str = "/jupyter-tutorial/hf_models/Qwen3-4B"):
        super().__init__(model=model_path)
        
        # Load the tokenizer and model (reusing your existing logic)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        
    def basic_request(self, prompt: str, **kwargs) -> dict:
        """Handle basic request for DSPy"""
        # Convert single prompt to messages format
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        # Tokenize
        model_inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        
        # Generate
        start_time = time.time()
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=kwargs.get('max_new_tokens', 512),
            pad_token_id=self.tokenizer.pad_token_id,
            temperature=kwargs.get('temperature', 0.1),
            top_p=kwargs.get('top_p', 0.9),
            do_sample=kwargs.get('do_sample', True)
        )
        generation_time = time.time() - start_time
        
        # Decode response
        input_ids = model_inputs.input_ids[0]
        generated_sequence = generated_ids[0]
        output_ids = generated_sequence[len(input_ids):].tolist()
        
        # Remove thinking tokens if present
        index = len(output_ids) - output_ids[::-1].index(151668) if 151668 in output_ids else 0
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
        
        # Return in the format expected by DSPy
        return {
            'choices': [{'message': {'content': content}}],
            'usage': {
                'total_tokens': len(output_ids),
                'generation_time': generation_time
            }
        }

class AnsweringAgent(object):
    r"""Agent responsible for answering MCQ questions with confidence scoring using DSPy"""
    
    def __init__(self, select_prompt1: bool = True, **kwargs):
        # Initialize DSPy with custom local Qwen LM
        lm = LocalQwenLM()
        dspy.configure(lm=lm)
        
        # Load DSPy program - assuming it's saved in a fixed location
        try:
            self.dspy_program = dspy.load("Evaluation/dspy/optimizer_scripts/dspy_programs/mcq_solver")
        except:
            # Create a simple DSPy program if loading fails
            self.dspy_program = self._create_default_program()
        
        self.select_prompt1 = select_prompt1
    
    def _create_default_program(self):
        """Create a default DSPy program for MCQ solving"""
        class MCQSolver(dspy.Module):
            def __init__(self):
                super().__init__()
                self.generate_answer = dspy.ChainOfThought("mcq_prompt -> expected_json_output")
            
            def forward(self, mcq_prompt):
                return self.generate_answer(mcq_prompt=mcq_prompt)
        
        return MCQSolver()
    
    def build_prompt(self, question_data: Dict[str, str|Any]) -> str:
        """Generate prompt for DSPy program"""
        
        instructions = (
            'INSTRUCTIONS FOR ANSWERING:\n'
            '1. Carefully read and understand what is being asked.\n'
            '2. Consider why each choice might be correct or incorrect.\n'
            '3. There is only **ONE OPTION** correct.\n'
            '4. Provide reasoning within 100 words\n\n'
        )
        
        question_text = f'Question: {question_data["question"]}\n'
        choices_text = f'Choices: {self._format_choices(question_data["choices"])}\n\n'
        
        format_instruction = (
            'RESPONSE FORMAT: Strictly generate a valid JSON object as shown below:\n'
            '{\n'
            '    "answer": "One of the letter from [A, B, C, D]",\n'
            '    "reasoning": "Brief explanation within 100 words"\n'
            '}'
        )
        
        return instructions + question_text + choices_text + format_instruction
    
    def answer_question(self, question_data: Dict|List[Dict], **kwargs) -> Tuple[List[str], int|None, float|None]:
        """Generate answer(s) for the given question(s) using DSPy"""
        if isinstance(question_data, list):
            responses = []
            total_tokens = 0
            total_time = 0
            
            for qd in question_data:
                prompt = self.build_prompt(qd)
                start_time = time.time()
                result = self.dspy_program(mcq_prompt=prompt)
                end_time = time.time()
                
                responses.append(result.expected_json_output)
                total_time += (end_time - start_time)
                # Note: Token counting would need to be implemented based on your needs
                
            return responses, total_tokens if total_tokens > 0 else None, total_time
        else:
            prompt = self.build_prompt(question_data)
            start_time = time.time()
            result = self.dspy_program(mcq_prompt=prompt)
            end_time = time.time()
            
            return result.expected_json_output, None, end_time - start_time
    
    def answer_batches(self, questions: List[Dict], batch_size: int = 5, **kwargs) -> Tuple[List[str], List[int | None], List[float | None]]:
        """Answer questions in batches"""
        answers = []
        tls, gts = [], []
        total_batches = (len(questions) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc="STEPS: ", unit="batch")
        
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]
            batch_answers, tl, gt = self.answer_question(batch_questions, **kwargs)
            answers.extend(batch_answers)
            tls.append(tl)
            gts.append(gt)
            pbar.update(1)
        
        pbar.close()
        return answers, tls, gts
    
    def count_tokens_a(self, text: str) -> int:
        """Count the number of tokens in the text using the LM's tokenizer"""
        lm = dspy.settings.lm
        if hasattr(lm, 'tokenizer'):
            return len(lm.tokenizer.encode(text, add_special_tokens=False))
        else:
            # Fallback to approximation
            return int(len(text.split()) * 1.3)
    
    def filter_answers(self, ans: List[str|Dict[str, str]]) -> List[Dict[str, str]]:
        r"""Filter answers to ensure they are in the correct format"""
        def basic_checks(a1: Dict[str, str])->bool:
            # check required keys
            required_keys = ['answer']
            if all((key in a1) and isinstance(a1[key], str) for key in required_keys):
                if len(a1['answer']) == 1 and (a1['answer'] not in 'ABCDabcd'):
                    return False
                check_len = self.count_tokens_a(a1['answer'])
                if check_len < 50:
                    check_len += self.count_tokens_a(a1.get('reasoning', 'None'))
                    if check_len < 512:
                        return True
            return False
    
        filtered_answers = []
        for i, a in enumerate(ans):
            if isinstance(a, dict):
                if basic_checks(a):
                    filtered_answers.append(a)
                else:
                    filtered_answers.append(None)
                    print(f"Skipping invalid answer at index {i}: {a}")
            elif isinstance(a, str):
                # Basic checks: at least with correct JSON format
                try:
                    a1 = json.loads(a)
                    if basic_checks(a1):
                        filtered_answers.append(a1)
                    else:
                        filtered_answers.append(None)
                        print(f"Skipping invalid answer at index {i}: {a}")
                except json.JSONDecodeError:
                    # If JSON decoding fails, skip this answer
                    print(f"Skipping invalid JSON at index {i}: {a}")
                    filtered_answers.append(None)
                    continue
            else:
                # If the answer is neither a dict nor a str, skip it
                print(f"Skipping unsupported type at index {i}: {type(a)}")
                filtered_answers.append(None)
        return filtered_answers
    
    def save_answers(self, answers: List[str], file_path: str|Path) -> None:
        """Save generated answers to a JSON file"""
        # check for existence of dir
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump([a for a in answers], f, indent=4)
    
    def _format_choices(self, choices: List[str]) -> str:
        r"""Format the choices for better readability"""
        formatted = []
        for choice in choices:
            # Ensure each choice starts with a letter if not already formatted
            if not re.match(r'^[A-D]\)', choice.strip()):
                # Extract letter from existing format or assign based on position
                letter = chr(65 + len(formatted))  # A, B, C, D
                formatted.append(f"{letter}) {choice.strip()}")
            else:
                formatted.append(choice.strip())
        return " ".join(formatted)

# Example usage
if __name__ == "__main__":
    import json
    import yaml
    import argparse
    from utils.build_prompt import auto_json, option_extractor_prompt
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # python -m agents.answer_agent --input_file outputs/filtered_questions.json --output_file outputs/answers.json --batch_size 5 --verbose
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    argparser = argparse.ArgumentParser(description="Run the Answering Agent with DSPy")
    argparser.add_argument("--input_file", type=str, default="outputs/filtered_questions.json", help="Path to the input JSON file with questions")
    argparser.add_argument("--output_file", type=str, default="outputs/answers.json", help="Path to save the answers")
    argparser.add_argument("--batch_size", type=int, default=5, help="Batch size for processing questions")
    argparser.add_argument("--verbose", action='store_true', help="Enable verbose output")
    args = argparser.parse_args()
    
    SELECT_PROMPT1 = False  # Use the first system prompt for answering
    
    # Load sample questions (assuming they're saved from QuestioningAgent)
    with open(args.input_file, 'r') as f:
        sample_questions = json.load(f)
    
    agent = AnsweringAgent(select_prompt1=SELECT_PROMPT1)
    
    # gen_kwargs = {"tgps_show": True, "max_new_tokens": 512, "temperature": 0.1, "top_p": 0.9, "do_sample": True}
    gen_kwargs = {"tgps_show": True}
    try:
        with open("agen.yaml", "r") as f: 
            gen_kwargs.update(yaml.safe_load(f))
    except FileNotFoundError:
        print("agen.yaml not found, using default parameters")
    
    answer, tls, gts = agent.answer_batches(
        questions=sample_questions,
        batch_size=args.batch_size,
        **gen_kwargs
    )
    
    ans = []
    for idx, (q, a) in enumerate(zip(sample_questions, answer)):
        if args.verbose:
            print(f"\n=== Question {idx+1} ===")
            print(f"Question: {q.get('question', 'N/A')}")
            print(f"Expected: {q.get('answer', 'N/A')}")
            print(f"Model Answer:\n{a}")
        
        try:
            if isinstance(a, str):
                a = json.loads(a)
            
            if isinstance(a, dict) and all(k in a for k in ['answer', 'reasoning']):
                # ++++++++++++++++++++++++++
                # TODO: IMPROVE THE FOLLOWING
                if len(a['answer']) != 1:
                    # Use DSPy program to extract option
                    extract_prompt = f"Extract only the letter option (A, B, C, or D) from: {a['answer']}"
                    result = agent.dspy_program(mcq_prompt=extract_prompt)
                    a['answer'] = result.expected_json_output
                # ++++++++++++++++++++++++++
            else:
                # the dictionary is not as expected. So extract it using DSPy
                prompt = (
                    'Extract **ONLY** the answer and reasoning while discarding the rest.\n\n'
                    'String:\n'
                    '{}\n\n'
                    'Given Format:\n'
                    '{{\n'
                    '    "answer": "Only the option letter (A, B, C, or D)",\n'
                    '    "reasoning": "..."\n'
                    '}}'
                )
                result = agent.dspy_program(mcq_prompt=prompt.format(json.dumps(a, indent=4)))
                a = result.expected_json_output
                
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error processing answer {idx}: {e}")
            # Use DSPy program to convert to proper JSON
            result = agent.dspy_program(mcq_prompt=f"Convert this to proper JSON format: {a}")
            a = result.expected_json_output
            
        ans.append(a)
        
    if args.verbose:
        if gen_kwargs.get('tgps_show', False):
            for idx, (tl, gt) in enumerate(zip(tls, gts)):
                print(f"BATCH - {idx}")
                if tl is not None:
                    print(f"Tokens: {tl}, Time: {gt:.3f} seconds")
                    print(f"TGPS: {tl/gt:.3f} tokens/second")
                else:
                    print(f"Time: {gt:.3f} seconds")
            print("\n" + "="*50)
            total_time = sum(gt for gt in gts if gt is not None)
            total_tokens = sum(tl for tl in tls if tl is not None)
            print(f"Total Time: {total_time:.3f} seconds")
            if total_tokens > 0:
                print(f"Total Tokens: {total_tokens}; TGPS: {total_tokens/total_time:.3f} tokens/second")
    
    # Save answers
    agent.save_answers(ans, args.output_file)
    filtered_file_name = args.output_file.replace("answers.json", "filtered_answers.json")
    agent.save_answers(agent.filter_answers(ans), filtered_file_name)