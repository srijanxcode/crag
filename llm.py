from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL = "google/flan-t5-base"

# Load tokenizer and model (CPU)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
model.eval()

def generate_answer(question, context):
    prompt = (
        "Using ONLY the information in the context below, answer the question.\n"
        "You may combine relevant facts from different parts of the context.\n"
        "If the context does not contain enough information, say \"I don't know\".\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    print("⚙️ LLM generating answer (flan-t5-base)...")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=90,
            do_sample=False,
            num_beams=2
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # ---------- SMART GUARDRAIL ----------
    # Reject useless fragment answers, allow short valid definitions
    too_short = len(answer.split()) < 5
    no_verb = not any(v in answer.lower() for v in [" is ", " was ", " are "])

    if too_short and no_verb:
        return "I don't know"

    return answer
def generate_web_answer(question, web_context):
    prompt = (
        "Using ONLY the web information below, answer the question.\n"
        "This answer is NOT based on uploaded documents.\n\n"
        f"Web Information:\n{web_context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    print("🌐 Generating WEB answer...")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=120,
            do_sample=False,
            num_beams=2
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
