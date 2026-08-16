import sys
import asyncio
sys.path.append(r'C:\Users\subha\OneDrive\Documents\CYRAX 3.0')
from orchestrator.planner import Planner


class DummyToolRegistry:
    def get_all_definitions(self):
        return [{"name": "TOOL"}]


class DummyBrainRouter:
    def __init__(self):
        self.calls = 0

    async def generate(self, messages=None, system_prompt=None, **kwargs):
        # First call returns malformed output, second call returns corrected JSON
        self.calls += 1
        if self.calls == 1:
            return 'Sorry, I cannot produce JSON. Here is some text.'
        return '{"thought": "done", "action": null, "action_args": {}, "final_answer": "Integration success"}'


class DummyCtx:
    def __init__(self):
        self.tool_registry = DummyToolRegistry()
        self.brain_router = DummyBrainRouter()


async def main():
    ctx = DummyCtx()
    planner = Planner(ctx=ctx, req_id="test:1", provider_name=None)
    result = await planner.run(user_input="Do something", history=[])
    print('Planner run result:', result)


if __name__ == '__main__':
    asyncio.run(main())
