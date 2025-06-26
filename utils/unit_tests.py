import tempfile
import subprocess

def run_unit_tests(code_snippet, test_cases,filepath):
    import re
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            code_file = f"{tmpdir}/{filepath}"

            with open(code_file, "w") as f:
                # Write the code
                f.write(code_snippet + "\n\n")

                # Define a mock `Pair` class (if needed)
                f.write("class Pair:\n")
                f.write("    def __init__(self, a, b):\n")
                f.write("        self.a = a\n")
                f.write("        self.b = b\n")
                f.write("    def __iter__(self):\n")
                f.write("        return iter((self.a, self.b))\n\n")

                # Write test cases as separate pytest test functions
                for i, assertion in enumerate(test_cases):
                    f.write(f"def test_case_{i}():\n")
                    f.write(f"    {assertion}\n\n")

        
            result = subprocess.run(
                ["pytest", "--tb=short", "-q", code_file],
                capture_output=True, text=True, timeout=10
            )
            output = result.stdout
            
            passed = failed = total = 0
            match = re.search(r"(\d+) passed", output)
            if match:
                passed = int(match.group(1))
            match = re.search(r"(\d+) failed", output)
            if match:
                failed = int(match.group(1))
            total = passed + failed
            pass_rate = passed / total if total > 0 else 0.0

            total = passed + failed
            pass_rate = passed / total if total > 0 else 0.0

            # print("#" * 50)
            # print(f"{len(test_cases)} test case(s) written.")
            # print(test_cases)
            # print("passed", passed, "failed", failed, "total", total, "pass_rate", pass_rate)
            # print("Output:\n", output)
            # print("#" * 50)

            return {
                "passed": passed,
                "failed": failed,
                "total": total,
                "pass_rate": pass_rate,
                "output": output
            }
    except subprocess.TimeoutExpired:
        return {
            "passed": 0,
            "failed": 0,
            "total": 0,
            "pass_rate": 0.0,
            "output": "Test execution timed out."
        }


