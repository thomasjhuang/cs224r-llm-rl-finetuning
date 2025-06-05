import re
import math

def parse_solution(solution_str):
    lines = solution_str.strip().split('\n')
    equation_line = None
    
    for line in reversed(lines):
        line = line.strip()
        if re.search(r"[0-9\.\s()\+\-\*\/]+=[0-9\.\s()]+$", line):
            equation_line = line
            break
    
    if not equation_line:
        for line in reversed(lines):
            if '=' in line:
                equation_line = line
                break
        if not equation_line:
            for line in reversed(lines):
                if line.strip():
                    equation_line = line
                    break
        if not equation_line:
            return None, "", "No equation found"

    match_eq = re.search(r"(.*?=[^\n\r=]*)", equation_line)
    if match_eq:
        equation_str = match_eq.group(1).strip()
    else:
        equation_str = equation_line

    return equation_str, equation_str, "Parsed"

def safe_eval(expression_str):
    allowed_chars = re.compile(r"^[0-9\s\.\+\-\*\/\(\)]*$")
    if not allowed_chars.match(expression_str):
        return None
    
    if re.search(r"/\s*0(?:\.0+)?(?![\.\d])", expression_str):
        return None

    try:
        return eval(expression_str)
    except:
        return None

def calculate_countdown_reward(generated_solution_str, numbers_available, target_number):
    format_score = 0.0
    verification_score = 0.0
    
    parsed_expr, raw_parsed, parse_feedback = parse_solution(generated_solution_str)

    if parsed_expr is None:
        return 0.0, f"Format Error: {parse_feedback}", 0.0, 0.0

    format_score = 0.1
    if raw_parsed:
        format_score = 0.25
    
    try:
        expr_part = parsed_expr.split('=')[0] if '=' in parsed_expr else parsed_expr
        number_matches = re.findall(r"\b\d*\.?\d+\b", expr_part)
        numbers_in_expr = [float(n) for n in number_matches]
    except:
        numbers_in_expr = []

    available_set = set(float(n) for n in numbers_available)
    expr_set = set(numbers_in_expr)
    
    numbers_used_correctly = expr_set.issubset(available_set)
    coverage = len(expr_set.intersection(available_set)) / len(available_set) if available_set else 0
    
    numbers_bonus = 0.0
    if numbers_used_correctly and coverage >= 0.75:
         numbers_bonus = 0.1
         format_score = 0.5

    calculated_result = None
    if '=' in parsed_expr:
        expr_to_eval = parsed_expr.split('=')[0].strip()
        claimed_result_str = parsed_expr.split('=')[1].strip()
        try:
            claimed_result = float(claimed_result_str)
        except:
            claimed_result = None 
    else:
        expr_to_eval = parsed_expr.strip()
        claimed_result = None

    if expr_to_eval:
        calculated_result = safe_eval(expr_to_eval)

    if calculated_result is None:
        verification_score += numbers_bonus * 0.5
        return format_score + verification_score, f"{parse_feedback}. Could not evaluate", format_score, verification_score

    if math.isclose(calculated_result, float(target_number)):
        verification_score = 0.25
        if numbers_bonus > 0:
            verification_score = 0.5 
        feedback = f"Correct! {parsed_expr} evaluates to {calculated_result}."
    else:
        verification_score = 0.05
        if numbers_bonus > 0:
             verification_score = 0.15
        feedback = f"Incorrect. {parsed_expr} evaluates to {calculated_result}, target was {target_number}."
        if claimed_result is not None and not math.isclose(claimed_result, calculated_result):
            feedback += f" Also claimed {claimed_result} which doesn't match."

    total_reward = format_score + verification_score
    return total_reward, feedback, format_score, verification_score


if __name__ == '__main__':
    reward, feedback, f_score, v_score = calculate_countdown_reward("8 + 5 = 13", [8, 5], 13)
    print(f"Basic test: {reward}")

    reward, feedback, f_score, v_score = calculate_countdown_reward("1 + 1 = 3", [1, 1], 2)
    print(f"Wrong: {reward}")
    
    reward, feedback, f_score, v_score = calculate_countdown_reward("blah", [1,2], 3)
    print(f"Garbage: {reward}")

    reward, feedback, f_score, v_score = calculate_countdown_reward("(8+5)*2-6 = 20", [8,5,2,6], 20)
    print(f"Complex: {reward}")
