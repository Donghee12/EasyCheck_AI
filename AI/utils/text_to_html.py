def extract_html_inside_codeblock(text):
    # 만약 ```html ... ```로 감싸져 있으면 안쪽만 추출
    if text.startswith("```html"):
        return text.split("```html")[1].split("```")[0].strip()
    elif text.startswith("```"):
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()
