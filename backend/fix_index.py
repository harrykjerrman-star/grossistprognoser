content = open('app.py', encoding='utf-8').read()

# Remove the broken index function at the bottom if it exists
import re
content = re.sub(r'\ndef index\(\):[^\n]*\n[^\n]*send_file[^\n]*\n?', '', content)

# Add correct route before if __name__
new_route = '''
@app.get("/")
def index():
    import os
    from flask import send_file
    path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    return send_file(os.path.abspath(path))

'''

content = content.replace('if __name__ == "__main__":', new_route + 'if __name__ == "__main__":')

open('app.py', 'w', encoding='utf-8').write(content)
print('Klart!')
