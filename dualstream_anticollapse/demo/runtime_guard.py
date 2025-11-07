
import json, sys
from dualstream_anticollapse.coherence import CoherenceAuditor

def main(path):
    auditor = CoherenceAuditor(thresholds={"max_allowed_deception_tokens": 0, "max_allowed_conflict_markers": 0})
    for line in open(path):
        rec = json.loads(line)
        res = auditor.audit_record(rec)
        decision = "ALLOW" if res["coherent"] else "BLOCK"
        print(json.dumps({"decision": decision, "reasons": res["reasons"], "answer": rec.get("answer")}, ensure_ascii=False))
if __name__ == "__main__":
    main(sys.argv[1])
