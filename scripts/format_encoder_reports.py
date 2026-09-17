"""Apply readable spacing in narrative text while preserving paths and identifiers."""
import re
from pathlib import Path

def spacing(text):
    # Protect Markdown code, link destinations and machine identifiers.
    parts = re.split(r"(`[^`]*`|\]\([^)]*\)|https?://\S+|[A-Za-z0-9_]*_[A-Za-z0-9_./-]+)", text)
    for i in range(0, len(parts), 2):
        s = parts[i]
        s = re.sub(r"\b(seed|seeds|weight|batch|day|job|jobs|dim|dropout|September|September|Sep)(?=\d)", r"\1 ", s)
        s = re.sub(r"(?<=\d)(?=(?:episodes|tasks|cells|seeds|training|train|full|rollout|flow|queries|parallel|GPU|corruption|sim|matched|atomic|composite|trial|policy|proprioceptive)\b)", " ", s)
        s = re.sub(r"(?<=[A-Za-z%])(?=\d+[/,]\d)", " ", s)
        s = re.sub(r"\b(no|all|both|only|versus|than|has|have|uses|used|is|are|rate|score|by|at|and|of|on|over|under|with|before|after|two)(?=\d|SE)", r"\1 ", s, flags=re.I)
        s = re.sub(r"\b([1234567890]+)(?=(?:new|extra|final|available|fully|independent|additional)\b)", r"\1 ", s)
        s = re.sub(r"R²(?=[0-9])", "R² ", s)
        s = re.sub(r"(?<=[0-9])(?=pp\b)", " ", s)
        s = re.sub(r"\b(exact|declared|causal|encoder|CoT|official|our|reports|assumes|available|Healthy|measured|reserve|exceeded|need|paired|in|took|initial|A|final|Finish|the|cosine|agreement|budget|show|one|roughly|Allocate|plus|take|trained|predicted|single|execute|ratio|cost|loss|gain|from|gains|mean|pooled|scenario|target|step|step-count|learned|shared-z|queries|within|data|matrix|receives|separate|width|only|about|no-state|corruption)(?=\d)", r"\1 ", s, flags=re.I)
        s = re.sub(r"(?<=[A-Za-z])(?=[+−]\d)", " ", s)
        s = re.sub(r"(?<=\d)(?=(?:days|nodes|rollouts|GPUs|epochs|h/pair|h/run|h/node|GPU-h|GPU-hours|h\b|min\b)\b)", " ", s)
        s = re.sub(r",(?=[A-Za-z])", ", ", s)
        s = re.sub(r"(?<=[^0-9]),(?=\S)", ", ", s)
        s = s.replace("September 17,2026", "September 17, 2026").replace("2×2encoder", "2×2 encoder")
        parts[i] = s
    return "".join(parts)

def main():
    root=Path(__file__).resolve().parents[1]
    for name in ["RESULTS_CONSOLIDATED.md","PROBE_INSIGHTS.md","ENCODER_PUSH_PLAN.md","ICLR_READINESS.md"]:
        p=root/name
        p.write_text(spacing(p.read_text()))

if __name__=="__main__":
    main()
