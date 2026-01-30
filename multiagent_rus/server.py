import json
import requests
from mcp.server.fastmcp import FastMCP
DEVDOCS_URL = "http://localhost:9292"

mcp = FastMCP("MegaSchool Server")


@mcp.tool()
def search_devdocs(doc_name: str, keyword: str) -> str:
    """
    Шаг 1. Ищет статьи. Возвращает slug и path для чтения.
    """
    try:
        manifest_url = f"{DEVDOCS_URL}/docs/docs.json"
        resp = requests.get(manifest_url, timeout=2)
        docs_list = resp.json()

        target_doc = next((d for d in docs_list if d['slug'].startswith(doc_name.lower())), None)
        if not target_doc:
            return f"Документация '{doc_name}' не найдена."

        slug = target_doc['slug']
        index_url = f"{DEVDOCS_URL}/docs/{slug}/index.json"
        idx_resp = requests.get(index_url, timeout=5)
        entries = idx_resp.json()['entries']

        results = []
        for entry in entries:
            if keyword.lower() in entry['name'].lower():
                results.append({
                    "title": entry['name'],
                    # Важно: возвращаем эти поля, чтобы LLM могла их использовать в след. шаге
                    "doc_slug": slug,
                    "path": entry['path'],
                    "url": f"{DEVDOCS_URL}/{slug}/{entry['path']}"
                })
                # if len(results) >= 5: break

        return json.dumps(results, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Ошибка поиска: {e}"


@mcp.tool()
def read_devdocs_page(doc_slug: str, path: str) -> str:
    """
    Шаг 2. Загружает полный HTML текст статьи.
    Args:
        doc_slug: slug документации (например 'python~3.14')
        path: путь к статье (например 'library/asyncio')
    """
    try:
        # 1. Загружаем базу данных конкретной доки (Внимание: файл может быть большим!)
        db_url = f"{DEVDOCS_URL}/docs/{doc_slug}/db.json"
        response = requests.get(db_url, timeout=10)

        if response.status_code != 200:
            return f"Ошибка загрузки базы данных: {response.status_code}"

        db_data = response.json()

        # 2. Достаем контент по ключу
        # Иногда path в индексе совпадает с ключом, иногда нет.
        # DevDocs хранит контент просто как строку HTML в значении ключа
        html_content = db_data.get(path)

        if not html_content:
            return "Статья не найдена в базе данных (проверьте path)."

        # 3. (Опционально) Можно почистить HTML от лишних тегов,
        # чтобы экономить токены. Но пока вернем как есть.
        # return html_content[:5000]  # Ограничим длину на всякий случай
        return html_content  # Ограничим длину на всякий случай

    except Exception as e:
        return f"Ошибка чтения статьи: {e}"


if __name__ == "__main__":
    mcp.run()
