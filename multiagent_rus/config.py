from langchain_community.chat_models import ChatOpenAI

DEV_DOCS_SLUGS = [
    # Languages
    "python", "go", "javascript", "typescript", "rust", "cpp", "c", "java",
    "kotlin", "php", "ruby", "swift", "dart", "elixir", "clojure", "lua",
    "perl", "scala", "haskell", "ocaml", "julia", "groovy", "crystal",
    "react", "vue", "svelte", "angular", "angularjs", "nextjs", "lit",
    "htmx", "jquery_core", "jquery_ui", "backbone", "ember", "marionette",
    "django", "flask", "fastapi", "rails", "laravel", "symfony", "express",
    "koa", "nestjs", "spring_boot", "django_rest_framework",
    "postgresql", "mysql", "mariadb", "sqlite", "redis", "influxdata",
    "docker", "kubernetes", "kubectl", "terraform", "ansible", "vagrant",
    "nginx", "haproxy", "git", "bash", "zsh", "fish", "nushell",
    "npm", "yarn", "bun", "deno", "node", "webpack", "vite", "esbuild",
    "eslint", "prettier", "jest", "playwright", "cypress", "puppeteer",
    "numpy", "pandas", "matplotlib", "scikit_learn", "tensorflow", "pytorch",
    "css", "html", "sass", "less", "tailwindcss", "bootstrap",
    "http", "svg", "dom", "web_extensions", "markdown", "latex",
]

TECH_SYNONYMS = {
    "golang": "go",
    "go": "go",
    "py": "python",
    "python": "python",
    "питон": "python",
    "го": "go",
}

DEBUG_RAG = True

llm = ChatOpenAI(
    model="meta-llama-3.1-8b-instruct",
    api_key="lm-studio",
    base_url="http://localhost:1234/v1",
    temperature=0.4,
)
