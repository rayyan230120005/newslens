"""
test_inference.py

Loads the fine-tuned checkpoint from train_bias_model.py and runs it on a
few example sentences, to sanity-check the saved model actually works
before wiring it into bias_service.py and the live API.

    python test_inference.py
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "bias-bert")

TEST_SENTENCES = [
    # Clearly BIASED - loaded language, framing, subjective adjectives
    "The minister slammed the so-called radical opposition in a shocking outburst.",
    "Extremist protesters violently disrupted the peaceful gathering, witnesses say.",
    "Officials said the controversial mayor ignored warnings before the bridge collapse.",
    "A fringe activist group staged a protest outside the courthouse and clashed with police.",
    "The senator called the proposal reckless and irresponsible without offering evidence.",
    "The corrupt politician shamelessly exploited the tragedy for political gain.",
    "Dangerous radicals stormed the building in a brazen attack on democracy.",
    "The incompetent administration utterly failed to address the crisis.",
    "Sinister forces behind the scenes are manipulating the narrative.",
    "The dictator's brutal regime continues its ruthless oppression of dissent.",

    # Clearly NOT_BIASED - factual, neutral, attribution-based
    "The city council approved the new budget in a 5-2 vote on Tuesday.",
    "The report states unemployment fell by 0.3 percent last quarter.",
    "The committee voted unanimously to extend the school funding plan for another year.",
    "The company reported a modest increase in quarterly revenue and stable operating costs.",
    "Researchers published a new study showing lower emissions after the policy change.",
    "The Supreme Court ruled 6-3 in favor of the plaintiff on Monday.",
    "Stock markets closed higher after the Federal Reserve announcement.",
    "The weather service issued a tornado warning for three counties.",
    "Scientists confirmed the vaccine shows 94% efficacy in clinical trials.",
    "The bridge reopened to traffic after safety inspections were completed.",

    # Edge cases - subtle framing, mixed signals
    "Critics argue the policy disproportionately affects low-income families.",
    "Supporters claim the measure will boost economic growth significantly.",
    "The legislation, which opponents call controversial, passed the Senate.",
    "Experts warn the new regulations could have unintended consequences.",
    "The governor's office declined to comment on the ongoing investigation.",
    "Both sides accused the other of spreading misinformation about the bill.",
    "Analysts suggest the tax reform may benefit corporations more than workers.",
    "The mayor announced a new initiative to address rising crime rates.",

    # News-style factual reporting with quotes
    "Police said the suspect was arrested without incident at 3 PM.",
    "The CEO stated the company remains committed to carbon neutrality by 2030.",
    "Witnesses reported hearing multiple gunshots near the downtown plaza.",
    "The health department confirmed 247 new cases in the past week.",
    "Firefighters contained the warehouse blaze after six hours.",

    # Opinion/editorial style (should be BIASED)
    "This disastrous policy will undoubtedly destroy countless jobs.",
    "Only a fool would believe the government's empty promises.",
    "The evidence overwhelmingly proves the administration is lying.",
    "It is abundantly clear that the opposition has no real solutions.",
    "Any reasonable person can see this proposal is fundamentally flawed.",
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    # Explicitly align the model's output classes to the dataset labels.
    if not hasattr(model.config, "id2label") or model.config.id2label in (None, {}):
        model.config.id2label = {0: "NOT_BIASED", 1: "BIASED"}
        model.config.label2id = {"NOT_BIASED": 0, "BIASED": 1}

    for text in TEST_SENTENCES:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = int(torch.argmax(probs).item())
        label = model.config.id2label.get(pred_idx, str(pred_idx))
        confidence = float(probs[pred_idx].item())

        print(f"\n[{label}] (confidence: {confidence:.2f})")
        print(f"  {text}")


if __name__ == "__main__":
    main()