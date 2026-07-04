"""The three enterprise deployment archetypes."""
import stratarag as sr
from stratarag.llm.echo import EchoProvider
from stratarag.orchestration import Orchestrator, Team, Workflow

kb = sr.Knowledge.from_texts(["Refund window is 14 days with receipt."])

# 1) Sequential workflow (e.g. touchless AP auditing)
wf = Workflow([
    ("ingest",    lambda t: f"line items extracted from {t}"),
    ("reconcile", sr.Agent(model=EchoProvider(script=["matches PO-991"]))),
    ("comply",    lambda t, s: f"APPROVED ({s['ingest'][:20]}... | {t})"),
])
print("workflow  :", wf.run("invoice_774.pdf").output)

# 2) Hub-and-spoke orchestrator (e.g. employee onboarding)
hub = Orchestrator({
    "billing": ("invoices refunds payments", sr.recipes.simple_rag(kb, "echo")),
    "it":      ("laptop hardware access accounts", lambda t: "laptop shipped"),
})
res = hub.run("customer wants a refund")
print("orchestr. :", res.state["routed_to"], "->", res.output[:50])

# 3) Collaborative team with critique (e.g. threat triage)
team = Team({
    "siem":      sr.Agent(model=EchoProvider(script=["30 failed logins from 3 ASNs"])),
    "forensics": sr.Agent(model=EchoProvider(script=["pattern matches MITRE T1110"])),
}, synthesizer=EchoProvider(script=["Brute-force attack confirmed; contain now."]))
print("team      :", team.run("investigate login anomalies").output)
