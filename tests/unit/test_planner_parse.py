from orchestrator.planner import Planner


def test_parse_various_formats():
    cases = [
        ('{"thought":"ok","action":null,"action_args":{},"final_answer":"done"}', True),
        ('```json\n{"thought":"ok","action":"TOOL","action_args":{},"final_answer":null}\n```', True),
        ('[{"thought":"ok","action":"TOOL","action_args":{"x":1},"final_answer":null}]', True),
        ('{"thought":"ok","actionArgs":{},"action":"TOOL","finalAnswer":null}', True),
        ('Some noise before {"thought":"hi","action":"TOOL","action_args":{},"final_answer":null} trailing', True),
        ('not a json at all', False),
        ('{"unexpected":"shape"}', False),
    ]

    for raw, should_succeed in cases:
        parsed = Planner._parse_decision(raw)
        if should_succeed:
            assert isinstance(parsed, dict), f"Expected success parsing: {raw}"
            assert (parsed.get('action') is not None) or (parsed.get('final_answer') is not None)
        else:
            assert parsed is None, f"Expected parse failure for: {raw}"
