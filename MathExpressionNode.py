"""
Math Expression Node for ComfyUI
================================
A powerful math expression evaluator with inputs a, b, c, d.
Supports all standard math operations, functions, and comparisons.

Author: ComfyUI-OllamaGemini
"""
import ast
import math
import random
import operator as op

# Hack: string type that is always equal in not equal comparisons
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")

# Supported operators
OPERATORS = {
    # Arithmetic
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg,
    ast.UAdd: lambda a: a,
    # Bitwise
    ast.BitXor: op.xor,
    ast.BitAnd: op.and_,
    ast.BitOr: op.or_,
    ast.Invert: op.invert,
    ast.RShift: op.rshift,
    ast.LShift: op.lshift,
    # Logical
    ast.And: lambda a, b: 1 if a and b else 0,
    ast.Or: lambda a, b: 1 if a or b else 0,
    ast.Not: lambda a: 0 if a else 1,
}

# Supported functions
FUNCTIONS = {
    # Rounding
    "round": lambda a, b=0: round(a, b),
    "ceil": math.ceil,
    "floor": math.floor,
    "trunc": math.trunc,
    # Min/Max
    "min": min,
    "max": max,
    "clamp": lambda x, lo, hi: max(lo, min(x, hi)),
    # Math
    "abs": abs,
    "sqrt": math.sqrt,
    "pow": pow,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    # Trigonometry
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "degrees": math.degrees,
    "radians": math.radians,
    # Hyperbolic
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    # Type conversion
    "int": int,
    "float": float,
    # Random
    "random": lambda: random.random(),
    "randint": random.randint,
    "uniform": random.uniform,
    "choice": lambda *args: random.choice(args),
    # Conditional
    "iif": lambda cond, t, f: t if cond else f,
    # Constants (as functions)
    "pi": lambda: math.pi,
    "e": lambda: math.e,
    "tau": lambda: math.tau,
}

def log(message, message_type='info'):
    prefix = "[MathExpr]"
    if message_type == 'error':
        print(f"{prefix} ERROR: {message}")
    else:
        print(f"{prefix} {message}")


class GeminiMathExpression:
    """
    Math Expression Evaluator
    -------------------------
    Evaluates mathematical expressions with inputs a, b, c, d.
    
    Supports:
    - Arithmetic: + - * / // % **
    - Bitwise: & | ^ ~ >> <<
    - Comparison: == != > < >= <=
    - Functions: sin, cos, sqrt, min, max, clamp, iif, etc.
    - Constants: pi(), e(), tau()
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expression": ("STRING", {
                    "default": "a + b",
                    "multiline": True,
                    "tooltip": """MATH EXPRESSION HELP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VARIABLES: a, b, c, d (connect any values)

ARITHMETIC OPERATORS:
  a + b    → Add
  a - b    → Subtract  
  a * b    → Multiply
  a / b    → Divide (float)
  a // b   → Divide (floor/int)
  a % b    → Modulo (remainder)
  a ** b   → Power (a^b)

COMPARISON (returns 1 or 0):
  a == b   → Equal
  a != b   → Not equal
  a > b    → Greater than
  a >= b   → Greater or equal
  a < b    → Less than
  a <= b   → Less or equal

BITWISE OPERATORS:
  a & b    → AND
  a | b    → OR
  a ^ b    → XOR
  ~a       → NOT (invert)
  a >> b   → Right shift
  a << b   → Left shift

LOGICAL (returns 1 or 0):
  a and b  → Logical AND
  a or b   → Logical OR
  not a    → Logical NOT

FUNCTIONS:
  min(a, b, ...)     → Minimum value
  max(a, b, ...)     → Maximum value
  clamp(x, lo, hi)   → Clamp x between lo and hi
  abs(a)             → Absolute value
  round(a, dp)       → Round to dp decimals
  floor(a)           → Round down
  ceil(a)            → Round up
  sqrt(a)            → Square root
  pow(a, b)          → Power (a^b)
  log(a)             → Natural logarithm
  log10(a)           → Base-10 logarithm
  exp(a)             → e^a
  
TRIGONOMETRY (radians):
  sin(a), cos(a), tan(a)
  asin(a), acos(a), atan(a)
  radians(deg)       → Convert degrees to radians
  degrees(rad)       → Convert radians to degrees

RANDOM:
  random()           → Random 0-1
  randint(a, b)      → Random int between a-b
  uniform(a, b)      → Random float between a-b
  choice(a, b, ...)  → Random choice

CONDITIONAL:
  iif(cond, t, f)    → If cond then t else f
  a if cond else b   → Python-style conditional

CONSTANTS:
  pi()  → 3.14159...
  e()   → 2.71828...
  tau() → 6.28318... (2*pi)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES:
  a + b * c           → Basic math
  max(a, b, c, d)     → Find maximum
  clamp(a, 0, 255)    → Limit to 0-255
  iif(a > b, a, b)    → Return larger value
  sqrt(a**2 + b**2)   → Pythagorean theorem
  sin(radians(a))     → Sine of angle in degrees
  round(a / b, 2)     → Divide and round to 2 decimals
  a if a > 0 else -a  → Absolute value
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""",
                }),
            },
            "optional": {
                "a": (any_type, {"default": 0, "tooltip": "Input variable 'a' - any numeric value"}),
                "b": (any_type, {"default": 0, "tooltip": "Input variable 'b' - any numeric value"}),
                "c": (any_type, {"default": 0, "tooltip": "Input variable 'c' - any numeric value"}),
                "d": (any_type, {"default": 0, "tooltip": "Input variable 'd' - any numeric value"}),
            },
        }
    
    RETURN_TYPES = ("INT", "FLOAT", "STRING")
    RETURN_NAMES = ("int", "float", "string")
    FUNCTION = "evaluate"
    CATEGORY = "AI API/Utils"
    
    @classmethod
    def IS_CHANGED(cls, expression, **kwargs):
        if "random" in expression.lower():
            return float("nan")
        return expression
    
    def evaluate(self, expression, a=0, b=0, c=0, d=0):
        # Clean expression
        expression = expression.replace('\n', ' ').replace('\r', '').strip()
        
        if not expression:
            return (0, 0.0, "0")
        
        # Convert inputs to numbers if possible
        def to_number(val):
            if val is None:
                return 0
            if isinstance(val, (int, float)):
                return val
            if isinstance(val, str):
                try:
                    return float(val) if '.' in val else int(val)
                except:
                    return 0
            # For tensors/images, could extract dimensions
            if hasattr(val, 'shape'):
                return val.shape[0]  # Return batch size
            return 0
        
        lookup = {
            "a": to_number(a),
            "b": to_number(b),
            "c": to_number(c),
            "d": to_number(d),
        }
        
        def eval_expr(node):
            # Constant (Python 3.8+)
            if isinstance(node, ast.Constant):
                return node.value
            
            # Binary operation: a + b, a * b, etc.
            elif isinstance(node, ast.BinOp):
                left = eval_expr(node.left)
                right = eval_expr(node.right)
                op_type = type(node.op)
                if op_type in OPERATORS:
                    return OPERATORS[op_type](left, right)
                raise NotImplementedError(f"Operator {op_type.__name__} not supported")
            
            # Unary operation: -a, ~a, not a
            elif isinstance(node, ast.UnaryOp):
                operand = eval_expr(node.operand)
                op_type = type(node.op)
                if op_type in OPERATORS:
                    return OPERATORS[op_type](operand)
                raise NotImplementedError(f"Unary operator {op_type.__name__} not supported")
            
            # Boolean operation: a and b, a or b
            elif isinstance(node, ast.BoolOp):
                values = [eval_expr(v) for v in node.values]
                op_type = type(node.op)
                result = values[0]
                for v in values[1:]:
                    result = OPERATORS[op_type](result, v)
                return result
            
            # Variable name: a, b, c, d
            elif isinstance(node, ast.Name):
                name = node.id.lower()
                if name in lookup:
                    return lookup[name]
                # Check if it's a constant function
                if name in FUNCTIONS:
                    fn = FUNCTIONS[name]
                    # For pi(), e(), tau() called without parens
                    if callable(fn):
                        try:
                            return fn()
                        except TypeError:
                            pass
                raise NameError(f"Unknown variable: {node.id}")
            
            # Function call: sin(a), max(a, b), etc.
            elif isinstance(node, ast.Call):
                func_name = node.func.id.lower() if isinstance(node.func, ast.Name) else str(node.func)
                if func_name not in FUNCTIONS:
                    raise NameError(f"Unknown function: {func_name}")
                
                args = [eval_expr(arg) for arg in node.args]
                return FUNCTIONS[func_name](*args)
            
            # Comparison: a > b, a == b, etc.
            elif isinstance(node, ast.Compare):
                left = eval_expr(node.left)
                result = 1
                for op_node, comparator in zip(node.ops, node.comparators):
                    right = eval_expr(comparator)
                    if isinstance(op_node, ast.Eq):
                        result = 1 if left == right else 0
                    elif isinstance(op_node, ast.NotEq):
                        result = 1 if left != right else 0
                    elif isinstance(op_node, ast.Gt):
                        result = 1 if left > right else 0
                    elif isinstance(op_node, ast.GtE):
                        result = 1 if left >= right else 0
                    elif isinstance(op_node, ast.Lt):
                        result = 1 if left < right else 0
                    elif isinstance(op_node, ast.LtE):
                        result = 1 if left <= right else 0
                    else:
                        raise NotImplementedError(f"Comparison {type(op_node).__name__} not supported")
                    if result == 0:
                        break
                    left = right
                return result
            
            # If expression: a if condition else b
            elif isinstance(node, ast.IfExp):
                condition = eval_expr(node.test)
                if condition:
                    return eval_expr(node.body)
                else:
                    return eval_expr(node.orelse)
            
            else:
                raise TypeError(f"Unsupported expression type: {type(node).__name__}")
        
        try:
            tree = ast.parse(expression, mode='eval')
            result = eval_expr(tree.body)
            
            # Convert result
            if isinstance(result, bool):
                result = 1 if result else 0
            
            int_result = int(result) if isinstance(result, (int, float)) else 0
            float_result = float(result) if isinstance(result, (int, float)) else 0.0
            str_result = str(result)
            
            return (int_result, float_result, str_result)
            
        except Exception as e:
            log(f"Expression error: {e}", 'error')
            return (0, 0.0, f"Error: {e}")


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeminiMathExpression": GeminiMathExpression,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiMathExpression": "Math Expression ➕",
}
