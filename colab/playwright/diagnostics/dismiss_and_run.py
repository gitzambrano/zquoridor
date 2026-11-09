import asyncio
from playwright.async_api import async_playwright

async def dismiss_and_run():
    async with async_playwright() as p:
        for port in [9001, 9002, 9003, 9004, 9005]:
            try:
                b = await p.chromium.connect_over_cdp(f'http://127.0.0.1:{port}')
                pg = b.contexts[0].pages[0]
                res = await pg.evaluate("""() => {
                    // Close dialogs
                    const dialogs = Array.from(document.querySelectorAll('mwc-dialog, md-dialog, .modal-dialog'));
                    let dismissed = 0;
                    for (const d of dialogs) {
                        try {
                            d.open = false;
                            d.style.display = 'none';
                            dismissed++;
                        } catch(e) {}
                    }
                    // Find run button
                    const btn = document.querySelector('colab-run-button');
                    let clicked = false;
                    if (btn) {
                        const shadowBtn = btn.shadowRoot ? btn.shadowRoot.querySelector('#run-button') : null;
                        const target = shadowBtn || btn;
                        target.click();
                        clicked = true;
                    }
                    return { dismissed, clicked };
                }""")
                print(f'{port}: {res}')
                await b.close()
            except Exception as ex:
                print(f'{port} err: {ex}')

asyncio.run(dismiss_and_run())
