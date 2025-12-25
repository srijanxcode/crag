def classify_failure(max_similarity, avg_similarity, answer):
    """
    Returns one of:
    - OK
    - OUT_OF_SCOPE
    - WEAK_CONTEXT
    - MODEL_UNCERTAIN
    """

    # Case 1: Documents irrelevant
    if max_similarity < 0.35:
        return "OUT_OF_SCOPE"

    # Case 2: Docs relevant but model refused
    if answer.strip().lower() == "i don't know":
        return "WEAK_CONTEXT"

    # Case 3: Model produced weak answer
    if len(answer.split()) < 6:
        return "MODEL_UNCERTAIN"

    return "OK"
