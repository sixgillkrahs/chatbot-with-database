from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import json

from app.core.config import GEMINI_KEY, ZOHO_MYSQL_DB, ZOHO_MYSQL_HOST, ZOHO_MYSQL_PASSWORD, ZOHO_MYSQL_PORT, ZOHO_MYSQL_USER
from app.prompts.sql_fewshots import SQL_FEWSHOTS
from app.schemas.InputChat import InputLangchain
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import FAISS
import threading
from app.services.embedding import embeddingModel
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
import urllib.parse
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_core.tools import BaseTool
from typing import List

_example_selector = None
_example_selector_lock = threading.Lock()

router = APIRouter(
    prefix="/langchain",
    tags=["langchain"]
)

# Custom toolkit to inject SafeQuerySQLDataBaseTool and exclude query checker
class SafeSQLDatabaseToolkit(SQLDatabaseToolkit):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_tools(self) -> List[BaseTool]:
        tools = super().get_tools()
        filtered_tools = []
        # for tool in tools:
        #     if tool.name == "sql_db_query_checker":
        #         continue
        #     if tool.name == "sql_db_query":
        #         safe_tool = SafeQuerySQLDataBaseTool(
        #             db=self.db,
        #             description=tool.description,
        #         )
        #         filtered_tools.append(safe_tool)
        #     else:
        #         filtered_tools.append(tool)
        return filtered_tools


def get_example_selector() -> SemanticSimilarityExampleSelector:
    global _example_selector
    if _example_selector is None:
        with _example_selector_lock:
            _example_selector = SemanticSimilarityExampleSelector.from_examples(
                examples=SQL_FEWSHOTS,
                k=3,
                vectorstore_cls=FAISS,
                embeddings=embeddingModel
            )
    return _example_selector

@router.get("/")
def read_root():
    return {"Hello": "World"}


def extract_output(event_data):
    if not event_data:
        return ""
        
    # First, try to get the nested output
    output = event_data.get("output")
    if output is None:
        # If output is not in "output" key, maybe it's in the root event_data itself
        output = event_data
        
    if isinstance(output, str):
        return output
        
    if isinstance(output, dict):
        # 1. Try common text keys
        for key in ["output", "result", "response", "answer"]:
            val = output.get(key)
            if val and isinstance(val, str):
                return val
                
        # 2. Try looking into messages list if present
        for key in ["messages", "chat_history"]:
            messages = output.get(key)
            if messages and isinstance(messages, list):
                last_msg = messages[-1]
                if hasattr(last_msg, "content"):
                    return str(last_msg.content)
                elif isinstance(last_msg, dict):
                    for content_key in ["content", "text", "message"]:
                        if last_msg.get(content_key):
                            return str(last_msg[content_key])
                else:
                    return str(last_msg)
                    
        # 3. If there is only one string value in the dict, return it
        str_vals = [v for v in output.values() if isinstance(v, str)]
        if len(str_vals) == 1:
            return str_vals[0]
            
        # 4. Fallback to json string
        return json.dumps(output, ensure_ascii=False)
        
    if isinstance(output, list):
        if output:
            last_item = output[-1]
            if hasattr(last_item, "content"):
                return str(last_item.content)
            elif isinstance(last_item, dict) and "content" in last_item:
                return str(last_item["content"])
            return str(last_item)
            
    return str(output) if output is not None else ""


async def event_generator(agent_executor, input_data):
    try:
        async for event in agent_executor.astream_events(
            {"input": input_data},
            version="v2"
        ):
            event_type = event.get("event")
            name = event.get("name")
            data = event.get("data", {})
            parent_ids = event.get("parent_ids", [])
            
            if event_type == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk is not None:
                    if hasattr(chunk, "content"):
                        content = chunk.content
                    elif isinstance(chunk, dict):
                        content = chunk.get("content", "")
                    else:
                        content = str(chunk)
                    
                    if content:
                        payload = {"event": "token", "data": str(content)}
                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                        
            elif event_type == "on_tool_start":
                payload = {
                    "event": "tool_start",
                    "tool": name,
                    "input": data.get("input")
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                
            elif event_type == "on_tool_end":
                payload = {
                    "event": "tool_end",
                    "tool": name,
                    "output": data.get("output")
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                
            elif event_type == "on_chain_end" and not parent_ids:
                final_text = extract_output(data)
                payload = {
                    "event": "final_result",
                    "output": final_text
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    except Exception as e:
        payload = {"event": "error", "message": str(e)}
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

@router.post("/chat")
async def chat(inputLangchain: InputLangchain):
    clearMessage = inputLangchain.userMessage.split("\n\n###")[0].strip()
    whitelisted_tables = [
        "purchase", "purchase_amount", "purchase_attachment", "purchase_detail", "purchase_info",
        "request_payment", "request_payment_attachment", "request_payment_info",
        "gruantee", "gruantee_attachment",
        "clearance_payment", "clearance_payment_attachment", "clearance_payment_info",
        "export_invoice", "export_invoice_goods",
        "synth_project", "contract_project", "bidding", "payslip"
    ]
    example_selector = get_example_selector()
    question_same = example_selector.select_examples({"question": clearMessage})
    few_shot_str = "\n\n".join(
        f"Question: {ex['question']}\nSQLQuery: {ex['sql_query']}"
        for ex in question_same
    )
    prefix = f"""Bạn là một nhà phân tích dữ liệu chuyên đổi các câu hỏi tiếng Việt thành truy vấn SQL và thực thi chúng trên cơ sở dữ liệu MySQL có tên "Zoho".

    Dưới đây là một số ví dụ:
    {few_shot_str}

    Hãy trả về kết quả bằng tiếng Việt."""
    password = urllib.parse.quote_plus(ZOHO_MYSQL_PASSWORD)
    db_uri = f"mysql+pymysql://{ZOHO_MYSQL_USER}:{password}@{ZOHO_MYSQL_HOST}:{ZOHO_MYSQL_PORT}/{ZOHO_MYSQL_DB}"
    db = SQLDatabase.from_uri(db_uri, include_tables=whitelisted_tables)
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.1,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=GEMINI_KEY
    )
    toolkit = SafeSQLDatabaseToolkit(db=db, llm=model)
    agent_executor = create_sql_agent(
        llm=model,
        toolkit=toolkit,
        agent_type="openai-tools",
        prefix=prefix,
        verbose=True,
        max_iterations=10,
        top_k=5,
    )

    return StreamingResponse(
        event_generator(agent_executor, clearMessage),
        media_type="text/event-stream"
    )