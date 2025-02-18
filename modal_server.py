import modal
import logging

# Get logger instance
logger = logging.getLogger(__name__)

# Don't override parent logger settings
logger.propagate = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Only log to console
    ]
)

web_api_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "beautifulsoup4",
    "colorama",
    "md2pdf",
    "python-dotenv",
    "pyyaml",
    "uvicorn",
    "pydantic==2.10.6",
    "fastapi",
    "python-multipart",
    "markdown",
    "langchain",
    "langchain_community",
    "langchain-openai",
    "langchain-ollama",
    "langgraph",
    "tiktoken",
    "gpt-researcher",
    "arxiv",
    "PyMuPDF",
    "requests",
    "jinja2",
    "aiofiles",
    "mistune",
    "python-docx",
    "htmldocx",
    "lxml_html_clean",
    "websockets",
    "unstructured",
    "json_repair",
    "json5",
    "loguru",
)

app = modal.App("gpt-researcher-server-2",
)
vol_output = modal.Volume.from_name("gpt-researcher-output", create_if_missing=True)
vol_docs = modal.Volume.from_name("gpt-researcher-docs", create_if_missing=True)

@app.function(
    secrets=[
        modal.Secret.from_name("anthropic"),
        modal.Secret.from_name("openai-secret"),
        modal.Secret.from_name("tavily_api"),
        modal.Secret.from_name("together_ai"),
        modal.Secret.from_dict({"MODAL_LOGLEVEL": "INFO"})
    ],
    volumes={"/outputs": vol_output, "/docs": vol_docs},
    container_idle_timeout=300,
    allow_concurrent_inputs=5,
    timeout=600,
    image=web_api_image,
)
@modal.asgi_app()
def web():
    import json
    import os
    from typing import Dict, List

    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, File, UploadFile, Header
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles

    from backend.server.websocket_manager import WebSocketManager
    from backend.server.server_utils import (
        handle_file_upload, handle_file_deletion,
        execute_multi_agents, handle_websocket_communication
    )
    from gpt_researcher import GPTResearcher

    # App initialization
    app = FastAPI()

    # WebSocket manager
    manager = WebSocketManager()

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/files/")
    async def list_files():
        files = os.listdir("/docs")
        print(f"Files in /docs: {files}")
        return {"files": files}


    @app.post("/api/multi_agents")
    async def run_multi_agents():
        return await execute_multi_agents(manager)


    @app.post("/upload/")
    async def upload_file(file: UploadFile = File(...)):
        return await handle_file_upload(file, DOC_PATH)


    @app.delete("/files/{filename}")
    async def delete_file(filename: str):
        return await handle_file_deletion(filename, DOC_PATH)


    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await manager.connect(websocket)
        try:
            await handle_websocket_communication(websocket, manager)
        except WebSocketDisconnect:
            await manager.disconnect(websocket)


    @app.post("/generate")
    async def get_report(request: Request) -> dict:
        body = await request.json()
        question = body.get("question")
        report_type = body.get("report_type")
        context = body.get("context")
        tone = body.get("tone")
        length = body.get("length")

        print(f"Request Body: {body}")
        combined_context = context
        if tone:
            combined_context += f"\n{tone}"
        if length:
            combined_context += f"Keep the writing to about {length} words."

        print(combined_context)

        researcher = GPTResearcher(question, report_type)
        researcher.set_verbose(True)
        research_result = await researcher.conduct_research()
        report = await researcher.write_report(ext_context=combined_context)

        source_urls = researcher.get_source_urls()
        research_costs = researcher.get_costs()

        return {
            "report": report,
            "source_urls": source_urls,
            "research_costs": research_costs,
        }

    app.mount("/outputs", StaticFiles(directory="/outputs"), name="outputs")
    app.mount("/docs", StaticFiles(directory="/docs"), name="docs")
    return app
