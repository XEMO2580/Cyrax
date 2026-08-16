import sys
sys.path.append(r'C:\Users\subha\OneDrive\Documents\CYRAX 3.0')
from orchestrator.planner import Planner

cases = [
    '{"thought":"ok","action":null,"action_args":{},"final_answer":"done"}',
    '```json\n{"thought":"ok","action":"TOOL","action_args":{},"final_answer":null}\n```',
    '[{"thought":"ok","action":"TOOL","action_args":{"x":1},"final_answer":null}]',
    '{"thought":"ok","actionArgs":{},"action":"TOOL","finalAnswer":null}',
    'Some noise before {"thought":"hi","action":"TOOL","action_args":{},"final_answer":null} trailing',
]

for i,c in enumerate(cases,1):
    res = Planner._parse_decision(c)
    print(i, res)
