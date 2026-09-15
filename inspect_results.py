import json

results = []

with open(
    "outputs/run_20260913_221653/hotpotqa/ollama/predictions.jsonl",
    encoding="utf-8"
) as f:

    for line in f:
        line = line.strip()

        if not line:
            continue

        obj = json.loads(line)

        if isinstance(obj, list):
            results.extend(obj)
        else:
            results.append(obj)


for i, r in enumerate(results, 1):

    print("\n" + "=" * 100)
    print("EXAMPLE", i)
    print("=" * 100)

    print("\nQUESTION:")
    print(r["question"])

    print("\nGOLD ANSWER:")
    print(r["gold_answers"])

    print("\nOLLAMA PREDICTION:")
    print(r["prediction"])

    print("\nSELECTED / COMPRESSED EVIDENCE:")

    selected = r.get("selected_sentences", [])

    if not selected:
        print("No selected_sentences field found.")
    else:
        for j, s in enumerate(selected, 1):
            if isinstance(s, dict):
                print("[" + str(j) + "]", s.get("text", s))
            else:
                print("[" + str(j) + "]", s)

    print("\nRETRIEVED DOCUMENTS:")

    docs = r.get("retrieved_documents", [])

    if not docs:
        print("No retrieved_documents field found.")
    else:
        for j, d in enumerate(docs, 1):
            if isinstance(d, dict):
                print(
                    "[" + str(j) + "]",
                    "score=", d.get("score"),
                    "rank=", d.get("rank")
                )
                print("    title:", d.get("title"))
                print("    text:", d.get("text", "")[:500])
            else:
                print("[" + str(j) + "]", d)

    print("\nMETRICS:")
    print("Exact Match:", r["exact_match"])
    print("F1:", r["f1"])
    print("Original tokens:", r["original_tokens"])
    print("Compressed tokens:", r["compressed_tokens"])