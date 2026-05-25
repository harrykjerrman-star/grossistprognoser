content = open('app.py', encoding='utf-8').read()
old = 'if __name__ == "__main__":\n    app.run(debug=True, port=5000)'
new = 'if __name__ == "__main__":\n    import os\n    port = int(os.environ.get("PORT", 5000))\n    app.run(debug=False, host="0.0.0.0", port=port)'
content = content.replace(old, new)
open('app.py', 'w', encoding='utf-8').write(content)
print('Klart!')
