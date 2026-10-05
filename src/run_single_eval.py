import csv

from .euri_rag import generate_answer


def get_first_question():
    with open(
        "data/evaluation/golden_dataset.csv",
        "r",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)
        return next(reader)


if __name__ == "__main__":
    row = get_first_question()

    question = row["question"]
    ground_truth = row["ground_truth"]

    answer = generate_answer(question)

    print("\nQuestion:")
    print(question)

    print("\nGround Truth:")
    print(ground_truth)

    print("\nGenerated Answer:")
    print(answer)