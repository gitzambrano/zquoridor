import asyncio
from playwright.async_api import async_playwright

async def inspect():
    async with async_playwright() as p:
        for port in [9002, 9003, 9004, 9005]:
            try:
                b = await p.chromium.connect_over_cdp(f'http://127.0.0.1:{port}')
                pg = b.contexts[0].pages[0]
                text = await pg.evaluate("""() => {
                    const outs = Array.from(document.querySelectorAll('.output-text, .stream, colab-output-renderer'));
                    return outs.map(o => o.innerText).filter(Boolean).slice(-2).join(' | ');
                }""")
                lines = [l.strip() for l in text.split('\n') if l.strip()]
                tail = lines[-1] if lines else 'NO_OUTPUT'
                print(f'{port}: {tail[:120]}')
                await b.close()
            except Exception as e:
                print(f'{port}: err {e}')

asyncio.run(inspect())
