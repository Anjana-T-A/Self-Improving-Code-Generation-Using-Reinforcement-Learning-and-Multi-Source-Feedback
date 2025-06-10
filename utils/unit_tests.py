import tempfile
import subprocess

def run_unit_tests(code_snippet, test_cases):
    """
    Executes unit tests against the provided code snippet.
    Args:
        code_snippet (str): Generated Python code.
        test_cases (str): Unit tests as a string.
    Returns:
        dict: {
            "passed": int,
            "failed": int,
            "total": int,
            "pass_rate": float
        }
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write the code snippet
        code_file = f"{tmpdir}/generated_code.py"
        with open(code_file, "w") as f:
            f.write(code_snippet + "\n")
            f.write(test_cases + "\n")
        
        # Run the unit tests using pytest
        try:
            result = subprocess.run(
                ["pytest", "--tb=short", "-q", code_file],
                capture_output=True, text=True, timeout=10
            )
            output = result.stdout
            
            # Parse pytest output
            passed = output.count("PASSED")
            failed = output.count("FAILED")
            total = passed + failed
            pass_rate = passed / total if total > 0 else 0.0
            
            return {
                "passed": passed,
                "failed": failed,
                "total": total,
                "pass_rate": pass_rate
            }
        except subprocess.TimeoutExpired:
            return {
                "passed": 0,
                "failed": 0,
                "total": 0,
                "pass_rate": 0.0
            }
