import uvicorn
import webbrowser
import threading
import time

def open_browser():
    time.sleep(1.5)
    print('\n>>> Opening dashboard in browser: http://127.0.0.1:8000/static/index.html <<<\n')
    webbrowser.open('http://127.0.0.1:8000/static/index.html')

if __name__ == '__main__':
    print('=' * 65)
    print('  CycloVision AI - National Cyclone Early Warning Command Center')
    print('  MoES / IMD Multi-Spectral Decision Support (SIH26070)')
    print('  Dashboard URL: http://127.0.0.1:8000/static/index.html')
    print('=' * 65)
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run('app:app', host='127.0.0.1', port=8000, reload=True)
