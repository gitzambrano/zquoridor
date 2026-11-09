import asyncio
from playwright.async_api import async_playwright

async def inspect():
    async with async_playwright() as p:
        for port in [9001, 9002, 9003, 9004, 9005]:
            try:
                b = await p.chromium.connect_over_cdp(f'http://127.0.0.1:{port}')
                pg = b.contexts[0].pages[0]
                state = await pg.evaluate("""() => {
                    const btn = document.querySelector('colab-run-button');
                    const dialogs = Array.from(document.querySelectorAll('mwc-dialog, md-dialog, .modal-dialog'));
                    const openDialog = dialogs.find(d => d.open || d.getAttribute('open') !== null || d.style.display !== 'none');
                    return {
                        title: document.title,
                        hasOpenDialog: !!openDialog,
                        dialogText: openDialog ? (openDialog.innerText || '').slice(0, 80) : '',
                        running: btn ? btn.getAttribute('running') : null
                    };
                }""")
                print(f'{port}: {state}')
                await b.close()
            except Exception as ex:
                print(f'{port} err: {ex}')

asyncio.run(inspect())
