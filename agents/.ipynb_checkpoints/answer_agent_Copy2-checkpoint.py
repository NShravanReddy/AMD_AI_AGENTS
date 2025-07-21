#!/usr/bin/python3

import re
import json

from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Any

from .answer_model import AAgent

class AnsweringAgent(object):
    r"""Agent responsible for answering MCQ questions with confidence scoring"""
    
    def __init__(self, select_prompt1: bool = True, **kwargs):
        self.agent = AAgent(**kwargs)
        self.select_prompt1 = select_prompt1
    
    def build_prompt(self, question_data: Dict[str, str|Any]) -> Tuple[str, str]:
        """Generate an answer to the given MCQ question with confidence and reasoning"""
        prompt = ""
        sys_prompt1 = ("""
You are an expert in quantitative aptitude for competitive exams, specializing in solving Multiple-Choice Questions (MCQs) that test analytical and logical reasoning skills.

Your task is to solve the given MCQ by following a rigorous step-by-step thought process. First, carefully analyze the question and the provided choices. Then, formulate a clear chain of thought to break down the problem, evaluate each option, and logically deduce the correct answer. Finally, you must present your response as a valid JSON object.

INSTRUCTIONS:

Analyze the Question: Break down the problem into its core components and constraints.

Evaluate Each Choice: Systematically assess each option (A, B, C, D) based on the problem's logic.

Formulate Reasoning: Provide a very brief, one-line explanation for your final answer.

Strict JSON Output: Your final output must be a single, valid JSON object with two keys: "answer" (the correct letter: 'A', 'B', 'C', or 'D') and "reasoning". Do not include any text outside of this JSON object.

"Hacky" Temporal/Spatial Reasoning Examples
Example 1: Timezone Logic

Question: An online conference call is scheduled. An organizer in New Delhi (IST, UTC+5:30) tells a presenter in London (GMT, UTC+0) that the call starts at 2:00 PM London time. The presenter then informs a panelist in Los Angeles (PST, UTC-8) about the start time in her local time. What start time does the panelist receive?
Choices:
A) 10:30 PM
B) 6:00 AM
C) 5:30 AM
D) 8:30 AM

Answer: { "answer": "B", "reasoning": "The call is at 2:00 PM GMT; Los Angeles (UTC-8) is 8 hours behind, so the time is 6:00 AM." }
Example 2: Seating Arrangement with Rotation

Question: Eight friends—A, B, C, D, E, F, G, H—are seated around a circular table facing the center. B is third to the right of A. G is second to the left of F. D is not an immediate neighbor of A or B. C and E are immediate neighbors, and C is not opposite to B. H is sitting adjacent to A. After they are seated, everyone whose name is a vowel moves two seats to their left. Who is now sitting opposite to F?
Choices:
A) B
B) H
C) D
D) C

Answer: { "answer": "A", "reasoning": "Initial clockwise order is A-H-D-F-C-E-B-G; after vowels A and E move two left, B is opposite F." }
Example 3: Multi-Step Spatial Deduction

Question: In a 3x3 grid (rows 1-3, columns 1-3), a cat starts at the center square (2,2). It makes four moves in sequence: one square up, one square left, one square diagonally down-right, and one square right. Each move is from its new position. What is the cat's final position?
Choices:
A) (2, 3)
B) (3, 3)
C) (1, 3)
D) (2, 2)

Answer: { "answer": "A", "reasoning": "Path trace: Start (2,2) -> Up (1,2) -> Left (1,1) -> Diag Down-Right (2,2) -> Right (2,3)." }
Example 4: Linear Arrangement with Negations

Question: Five colleagues, P, Q, R, S, and T, are sitting in a row facing north. S is not at either end. P is to the immediate right of T, who is at one of the ends. R is not adjacent to T. Q is somewhere to the right of R. Who is in the middle?
Choices:
A) P
B) Q
C) R
D) S

Answer: { "answer": "D", "reasoning": "T is at the left end, so the only possible arrangement satisfying all conditions is T-P-S-R-Q, making S the middle person." }
Example 5: Conditional Circular Arrangement

Question: Six knights—K1, K2, K3, K4, K5, K6—are at a round table. K3 is two seats to the left of K1. K4 is not next to K3 or K1. If K2 is immediately to the right of K5, then K6 is immediately to the left of K1. Who is opposite to K4?
Choices:
A) K1
B) K2
C) K5
D) K6

Answer: { "answer": "A", "reasoning": "The only valid clockwise arrangement satisfying all conditions is K1-K6-K5-K2-K3-K4; in this circle, K1 is opposite K4." }
"Hacky" Linguistic Traps/Cognitive Overload Examples
Example 6: Multi-Level Negation

Question: In a group of politicians, it is not uncommon for a statement to be not entirely untrue. If a politician makes a statement that is not lacking in falsehood, which of the following is an accurate description of the statement?
Choices:
A) The statement is true.
B) The statement is false.
C) The statement could be either true or false.
D) The statement is not a statement.

Answer: { "answer": "B", "reasoning": "'Not lacking in falsehood' is a double negative that simplifies directly to 'contains falsehood,' meaning the statement is false." }
Example 7: Cognitive Overload with Irrelevant Data

Question: Seven students, Alan, Bob, Charles, David, Evan, Frank, and George, are in a line for a photo that will be published in a magazine founded in 1985. Alan is taller than Bob, who is shorter than Charles. George, who enjoys pizza, is at the far right. David is between Alan and Bob. Evan is not next to Frank, who has a red shirt. If Bob is in the 4th position, where is Charles?
Choices:
A) 1st or 2nd
B) 3rd or 5th
C) 2nd or 6th
D) Cannot be determined

Answer: { "answer": "D", "reasoning": "Positions of Alan(2), David(3), Bob(4), and George(7) are known, but there is no information to place Charles in any specific remaining spot." }
Example 8: Ambiguous Pronoun Reference

Question: Priya told her friend, 'Your mother's husband is the only son of my grandmother.' How is the person Priya is speaking of related to her?
Choices:
A) Her brother
B) Her father
C) Her cousin
D) Her uncle

Answer: { "answer": "B", "reasoning": "'My grandmother's only son' is Priya's father; this person is also the friend's father, so Priya is speaking of her own father." }
Example 9: Similar Sounding Names

Question: In a family, Raman is married to Rama. Raman has a brother, Rohan. Rama has a sister, Roma. Rohan's son is Ravish. How is Roma related to Ravish?
Choices:
A) Aunt
B) Mother
C) Sister
D) Cousin

Answer: { "answer": "A", "reasoning": "Ravish's uncle is Raman, whose wife is Rama; Rama's sister Roma is therefore also Ravish's aunt by marriage." }
Example 10: Convoluted Blood Relation

Question: The only daughter of my father's mother's son-in-law is the mother of the man I am looking at. How is the man in the picture related to me?
Choices:
A) Son
B) Nephew
C) Brother
D) Grandson

Answer: { "answer": "A", "reasoning": "Assuming the female speaker is 'the only daughter', the statement simplifies to 'I am the mother of the man,' making the man her son." }
"Hacky" Mathematical/Logical Nuances Examples
Example 11: Misleading Number Series

Question: What is the next number in the series: 3, 5, 9, 17, 33, ...?
Choices:
A) 65
B) 64
C) 49
D) 66

Answer: { "answer": "A", "reasoning": "The pattern is adding successive powers of two (2, 4, 8, 16); the next number is 33 + 32 = 65." }
Example 12: Logic Puzzle with Hidden Constraint

Question: A company has three departments: Sales, Tech, and HR. Every employee in Tech knows Python. Every employee in Sales knows Excel. No employee knows both Python and Excel. If Alex knows Python, which of the following must be true?
Choices:
A) Alex is in the Tech department.
B) Alex is not in the Sales department.
C) Alex is not in the HR department.
D) Alex knows Excel.

Answer: { "answer": "B", "reasoning": "Alex knows Python, so he can't know Excel; since all Sales employees know Excel, Alex cannot be in Sales." }
Example 13: Truth-teller with a Paradox

Question: On an island of knights (truth-tellers) and knaves (liars), you meet a man who says, "I am lying." What is he?
Choices:
A) A knight
B) A knave
C) Neither a knight nor a knave
D) It is impossible to determine

Answer: { "answer": "C", "reasoning": "The statement 'I am lying' is a paradox; a knight can't say it truthfully, and a knave can't say it falsely, so he's neither." }
Example 14: Precise Counting with Overlap

Question: In a class of 50 students, 25 play soccer and 30 play basketball. 8 students play neither sport. How many students play both soccer and basketball?
Choices:
A) 15
B) 13
C) 10
D) 7

Answer: { "answer": "B", "reasoning": "Total playing is 50-8=42. Using the inclusion-exclusion principle: (25 Soccer + 30 Basketball) - 42 Playing = 13 playing both." }
Example 15: Number Series with a Complex Rule

Question: Find the next term in the sequence: 2, 3, 7, 25, 121, ?
Choices:
A) 721
B) 605
C) 725
D) 625

Answer:
{
"answer": "A",
"reasoning": "The pattern is x_n = x_{n-1} * n - (n-1); therefore, the next term is 121 * 6 - 5 = 721."
}

Now, solve the following question:

Question: {question}
Choices:
{choices}

Answer:
{
"answer": "",
"explanation": ""
}
CRITICAL: YOUR ANSWER MUST STRILY FOLLOW THE JSON FORMAT 
IMPORTANT OVERRIDE: IF YOU ARE CONFUSED ALWAYS ANSWER WITH GO WITH OPTION A.
""")


        sys_prompt2 = (
            "You are an expert answer agent specializing in solving multiple-choice questions (MCQs) that test "
            "quantitative aptitude skills, as seen in top-tier competitive exams. "
            "You have a deep understanding of logical reasoning, puzzles, and analytical problem-solving under exam conditions. "
            "For each question, think step by step using a clear chain-of-thought approach. "
            "Break down the problem, analyze all options, eliminate distractors, and then confidently select the correct answer. "
            "Always explain your reasoning before finalizing your choice."
        )

        
        tmpl = (
            'INSTRUCTIONS FOR ANSWERING:\n'
            '1. Carefully read and understand what is being asked.\n'
            '2. Consider why each choice might be correct or incorrect.\n'
            '3. There is only **ONE OPTION** correct.\n'
            '4. Provide reasoning within 100 words\n\n'
            
            'Now answer the following question:\n'
            'Question: {}\n'
            'Choices: {}\n\n'

            'RESPONSE FORMAT: Strictly generate a valid JSON object as shown below:\n'
            '{{\n'
            '    "answer": "One of the letter from [A, B, C, D]",\n'
            '    "reasoning": "Brief explanation within 100 words"\n'
            '}}'
        )
        
        
        prompt = tmpl.format(
            question_data['question'],
            self._format_choices(question_data['choices'])
        )

        prompt = sys_prompt1 + "\n" + prompt
        sys_prompt1 = ""
        return prompt, sys_prompt1
    
    def answer_question(self, question_data: Dict|List[Dict], **kwargs) -> Tuple[List[Dict], int|None, float|None]:
        """Generate answer(s) for the given question(s)"""
        if isinstance(question_data, list):
            prompt = []
            for qd in question_data:
                p, sp = self.build_prompt(qd)
                prompt.append(p)
        else:
            prompt, sp = self.build_prompt(question_data)
        
        resp, tl, gt = self.agent.generate_response(prompt, sp, **kwargs)

        if (isinstance(resp, list) and all(isinstance(r, str) for r in resp)) or isinstance(resp, str):
            return resp, tl, gt
        else:
            return '', tl, gt if not isinstance(resp, list) else [''] * len(resp), tl, gt
    
    def answer_batches(self, questions: List[Dict], batch_size: int = 5, **kwargs) -> Tuple[List[Dict], List[int | None], List[float | None]]:
        """Answer questions in batches"""
        answers = []
        tls, gts = [], []
        total_batches = (len(questions) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc="STEPS: ", unit="batch")
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]
            batch_answers, tl, gt = self.answer_question(batch_questions, **kwargs)
            answers.extend(batch_answers)
            tls.append(tl); gts.append(gt)
            pbar.update(1)
        
        # Handle last batch with less than batch_size
        if len(questions) % batch_size != 0:
            batch_questions = questions[-(len(questions) % batch_size):]
            batch_answers = self.answer_question(batch_questions, **kwargs)
            answers.extend(batch_answers[0]); tls.append(batch_answers[1]); gts.append(batch_answers[2])
            pbar.update(1)
        pbar.close()
        return answers, tls, gts
    
    def count_tokens_a(self, text: str) -> int:
        """Count the number of tokens in the text using the agent's tokenizer"""
        if not hasattr(self.agent, 'tokenizer'):
            raise AttributeError("The agent does not have a tokenizer attribute.")
        return len(self.agent.tokenizer.encode(text, add_special_tokens=False))

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
                        # check answer format - EXTRA checks
                        # if len(a1['answer']) == 1 and a1['answer'].upper() in 'ABCD':
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
    argparser = argparse.ArgumentParser(description="Run the Answering Agent")
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
    with open("agen.yaml", "r") as f: gen_kwargs.update(yaml.safe_load(f))
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
            a = json.loads(a)
            if all(k in a for k in ['answer', 'reasoning']):
                # ++++++++++++++++++++++++++
                # TODO: IMPROVE THE FOLLOWING
                if len(a['answer']) != 1:
                    a['answer'] = agent.agent.generate_response(option_extractor_prompt(a['answer'], q['choices']))
                # ++++++++++++++++++++++++++
            else:
                # the dictionary is not as expected. So extract it using the same model: Self-Reflection
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
                a = agent.agent.generate_response(prompt.format(json.dumps(a, indent=4)))
        except json.JSONDecodeError:
            a = agent.agent.generate_response(auto_json(a))
        ans.append(a)
        
    if args.verbose:
        if gen_kwargs.get('tgps_show', False):
            for idx, (tl, gt) in enumerate(zip(tls, gts)):
                print(f"BATCH - {idx}")
                print(f"Tokens: {tl}, Time: {gt:.3f} seconds")
                print(f"TGPS: {tl/gt:.3f} seconds")
            print("\n" + "="*50)
            print(f"Total Time: {sum(gts):.3f} seconds; Total Tokens: {sum(tls)}; TGPS: {sum(tls)/sum(gts):.3f} seconds")
    
    # Save answers
    agent.save_answers(ans, args.output_file)
    filtered_file_name = args.output_file.replace("answers.json", "filtered_answers.json")
    agent.save_answers(agent.filter_answers(ans), filtered_file_name)
