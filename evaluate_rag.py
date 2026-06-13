import pandas as pd
from rag_chain import build_chain

# Load evaluation dataset
df = pd.read_csv("eval_dataset.csv")

# Load RAG chain
chain = build_chain()

results = []

print("Starting evaluation...\n")

for idx, row in df.iterrows():

    question = row["question"]
    expected_answer = row["expected_answer"]
    source = row["source"]

    print(f"[{idx+1}/{len(df)}] {question}")

    try:
        generated_answer = chain.invoke(question)

        results.append({
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer,
            "source": source
        })

    except Exception as e:

        results.append({
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": f"ERROR: {e}",
            "source": source
        })

# Save results
results_df = pd.DataFrame(results)

results_df.to_csv(
    "evaluation_results.csv",
    index=False
)

print("\nDone!")
print("Results saved to evaluation_results.csv")