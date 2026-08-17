# scripts/provider_smoke.py
import asyncio
import logging
from brain.providers.groq_provider import GroqProvider
from brain.providers.gemini_provider import GeminiProvider
from brain.moe_router import MoERouter
from brain.provider_metrics import ProviderMetricsManager
from core.resource_manager import ResourceManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('smoke')

async def run():
    # Instantiate providers (boot already ran in prior step but test directly)
    try:
        groq = GroqProvider()
    except Exception as exc:
        logger.error(f'Groq init failed: {exc}')
        groq = None

    try:
        gemini = GeminiProvider()
    except Exception as exc:
        logger.error(f'Gemini init failed: {exc}')
        gemini = None

    providers = {}
    if groq:
        providers['groq'] = groq
    if gemini:
        providers['gemini'] = gemini

    metrics = ProviderMetricsManager()
    rm = ResourceManager()
    router = MoERouter(providers=providers, metrics_manager=metrics, resource_manager=rm)

    # Print capability matrix
    for name, p in providers.items():
        caps = getattr(p, 'capabilities', None)
        logger.info(f'Provider={name} capabilities={caps.__dict__ if caps else None}')

    # Helper to attempt a generate with timeout
    async def try_generate(provider_name, gen_mode, json_mode=False):
        try:
            if provider_name == 'direct_groq' and groq:
                coro = groq.generate(messages=[{'role':'user','content':'Hello'}], system_prompt='sys', max_tokens=50, generation_mode=gen_mode, json_mode=json_mode)
            elif provider_name == 'direct_gemini' and gemini:
                coro = gemini.generate(messages=[{'role':'user','content':'Hello'}], system_prompt='sys', max_tokens=50, generation_mode=gen_mode, json_mode=json_mode)
            else:
                # Use router to respect failover/capabilities
                coro = router.generate(messages=[{'role':'user','content':'Hello'}], system_prompt='sys', generation_mode=gen_mode, provider_name=(provider_name if provider_name in providers else None), intent='smoke', trace_id='smoke-test')
            res = await asyncio.wait_for(coro, timeout=10)
            logger.info(f'{provider_name} {gen_mode} -> OK, len={len(res) if isinstance(res,str) else type(res)}')
        except Exception as exc:
            logger.error(f'{provider_name} {gen_mode} -> FAIL: {exc}')

    # Run smoke attempts
    await try_generate('direct_groq', 'text')
    await try_generate('direct_gemini', 'text')
    await try_generate('groq', 'structured_json')
    await try_generate('gemini', 'text')

if __name__ == '__main__':
    asyncio.run(run())
