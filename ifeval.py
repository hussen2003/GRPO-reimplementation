from deepeval.benchmarks import IFEval, GSM8K
from deepeval.models import OllamaModel
import json
from datetime import datetime

custom_model_name = "grpo" 
output_filename = "ifeval_report.txt"

my_ollama_model = OllamaModel(
    model=custom_model_name,
    base_url="http://localhost:11434",
)

benchmark = IFEval(n_problems=1) #change to IFEval or GSM8K

print(f"Starting IFEval benchmark on {custom_model_name}...")

benchmark_result = benchmark.evaluate(model=my_ollama_model) 

# Access the final score from the benchmark object 
overall_score = benchmark.overall_score
print(overall_score)

# Get the full report JSON from the captured result object

with open(output_filename, "a") as f:
    f.write(f"Model: {custom_model_name}\n")
    f.write(f"IFEval Score: {overall_score:.4f}\n")
    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("-" * 30 + "\n")

print(f"\n Evaluation complete. Full report written to {output_filename}")
