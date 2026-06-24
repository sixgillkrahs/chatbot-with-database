from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import json
import re
import aiomysql
from typing import List, Optional, Any, Dict

from app.core.config import (
    GEMINI_KEY,
    OPENROUTER_KEY, 
    ZOHO_MYSQL_DB, 
    ZOHO_MYSQL_HOST, 
    ZOHO_MYSQL_PASSWORD, 
    ZOHO_MYSQL_PORT, 
    ZOHO_MYSQL_USER
)
from app.schemas.InputChat import InputLangchain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import BaseTool

router = APIRouter(
    prefix="/handle",
    tags=["handle"]
)

# --- Database Pool Manager (Reused from MCP principles) ---
db_pool = None

async def get_db_pool():
    global db_pool
    if db_pool is None:
        db_pool = await aiomysql.create_pool(
            host=ZOHO_MYSQL_HOST,
            port=int(ZOHO_MYSQL_PORT or 3306),
            user=ZOHO_MYSQL_USER,
            password=ZOHO_MYSQL_PASSWORD,
            db=ZOHO_MYSQL_DB,
            minsize=1,
            maxsize=10,
            autocommit=True
        )
    return db_pool

async def execute_query(sql: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
    """
    Executes a SQL query safely, mirroring the security rules of the MCP server:
    1. Removes comments.
    2. Enforces read-only query prefixes.
    3. Blocks LOAD_FILE() and INTO OUTFILE/DUMPFILE injection vectors.
    """
    pool = await get_db_pool()
    
    # 1. Strip SQL comments (single-line and multi-line)
    sql_no_comments = re.sub(r'--.*?$', '', sql, flags=re.MULTILINE)
    sql_no_comments = re.sub(r'/\*.*?\*/', '', sql_no_comments, flags=re.DOTALL)
    sql_no_comments = sql_no_comments.strip()
    
    query_upper = sql_no_comments.upper()
    
    # 2. Read-only validation (Allow SELECT, SHOW, DESC, DESCRIBE, USE, EXPLAIN)
    allowed_prefixes = ('SELECT', 'SHOW', 'DESC', 'DESCRIBE', 'USE', 'EXPLAIN')
    is_allowed = any(query_upper.startswith(prefix) for prefix in allowed_prefixes)
    if not is_allowed:
        raise PermissionError("Thao tác bị từ chối: Chỉ cho phép thực hiện các câu lệnh đọc dữ liệu (SELECT, SHOW, DESCRIBE, EXPLAIN).")
        
    # 3. Check for LOAD_FILE() function (case-insensitive, outside strings)
    sql_no_strings = re.sub(r"'(?:[^'\\]|\\.)*'", "''", sql_no_comments)
    sql_no_strings = re.sub(r'"(?:[^"\\]|\\.)*"', '""', sql_no_comments)
    sql_no_strings_upper = sql_no_strings.upper()
    
    if re.search(r'\bLOAD_FILE\s*\(', sql_no_strings_upper):
        raise PermissionError("Thao tác bị từ chối: Không được phép sử dụng hàm LOAD_FILE() vì lý do bảo mật.")
        
    # 4. Check for SELECT ... INTO OUTFILE/DUMPFILE
    if re.search(r'\bINTO\s+(OUTFILE|DUMPFILE)\b', sql_no_strings_upper):
        raise PermissionError("Thao tác bị từ chối: Không được phép ghi file bằng INTO OUTFILE hoặc INTO DUMPFILE.")

    # 5. Execute query using the pool connection
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            if params:
                await cursor.execute(sql, params)
            else:
                await cursor.execute(sql)
            result = await cursor.fetchall()
            return result

# --- Custom Langchain Tools Wrapping MCP Logic ---

@tool
async def list_tables() -> str:
    """Lists all available tables in the database to understand what data exists."""
    try:
        results = await execute_query("SHOW TABLES")
        tables = [list(row.values())[0] for row in results]
        return f"Các bảng có sẵn trong database: {', '.join(tables)}"
    except Exception as e:
        return f"Lỗi khi liệt kê danh sách bảng: {str(e)}"

@tool
async def get_table_schema(table_name: str) -> str:
    """Retrieves the schema (column names, types, nullability, keys, default values) for a specific table.
    Use this to inspect columns before writing SQL queries.
    """
    if not table_name.isidentifier():
        return "Tên bảng không hợp lệ."
    try:
        results = await execute_query(f"DESCRIBE `{table_name}`")
        schema_info = {}
        for row in results:
            col_name = row.get('Field')
            if col_name:
                schema_info[col_name] = {
                    'type': row.get('Type'),
                    'nullable': row.get('Null') == 'YES',
                    'key': row.get('Key'),
                    'default': row.get('Default'),
                    'extra': row.get('Extra')
                }
        return f"Schema của bảng `{table_name}`:\n{json.dumps(schema_info, ensure_ascii=False, indent=2)}"
    except Exception as e:
        return f"Lỗi khi lấy schema của bảng {table_name}: {str(e)}"

@tool
async def get_table_schema_with_relations(table_name: str) -> str:
    """Retrieves the schema for a specific table, including foreign key relationships and referenced tables.
    Use this to understand how tables link together.
    """
    if not table_name.isidentifier():
        return "Tên bảng không hợp lệ."
    try:
        basic_schema_str = await get_table_schema(table_name)
        
        fk_sql = """
        SELECT 
            kcu.COLUMN_NAME as column_name,
            kcu.REFERENCED_TABLE_NAME as referenced_table,
            kcu.REFERENCED_COLUMN_NAME as referenced_column
        FROM information_schema.KEY_COLUMN_USAGE kcu
        WHERE kcu.TABLE_SCHEMA = %s 
          AND kcu.TABLE_NAME = %s 
          AND kcu.REFERENCED_TABLE_NAME IS NOT NULL
        """
        fk_results = await execute_query(fk_sql, (ZOHO_MYSQL_DB, table_name))
        
        fk_info = []
        for row in fk_results:
            fk_info.append(f"- Cột `{row['column_name']}` liên kết tới bảng `{row['referenced_table']}` cột `{row['referenced_column']}`")
            
        relation_str = "\n\nMối quan hệ khóa ngoại:\n" + "\n".join(fk_info) if fk_info else "\n\nBảng này không có khóa ngoại nào."
        return f"{basic_schema_str}{relation_str}"
    except Exception as e:
        return f"Lỗi khi lấy thông tin quan hệ của bảng {table_name}: {str(e)}"

@tool
async def execute_sql(sql_query: str) -> str:
    """Executes a read-only SQL query against the database and returns the results.
    Enforces security by rejecting modifying statements (INSERT, UPDATE, DELETE, etc.).
    """
    try:
        results = await execute_query(sql_query)
        if not results:
            return "Truy vấn thành công nhưng không có bản ghi nào được trả về."
            
        # Limit results size to avoid token bloat
        max_rows = 50
        limited_results = results[:max_rows]
        output = f"Kết quả truy vấn ({len(limited_results)} dòng đầu tiên):\n{json.dumps(limited_results, ensure_ascii=False, indent=2)}"
        if len(results) > max_rows:
            output += f"\n\n(Lưu ý: Kết quả thực tế có {len(results)} dòng, hệ thống tự động giới hạn hiển thị {max_rows} dòng)"
        return output
    except Exception as e:
        return f"Lỗi thực thi SQL: {str(e)}"

# --- Output Extractor & Streaming Logic (Reused from langchain.py) ---

def extract_output(event_data):
    if not event_data:
        return ""
        
    output = event_data.get("output")
    if output is None:
        output = event_data
        
    if isinstance(output, str):
        return output
        
    if isinstance(output, dict):
        for key in ["output", "result", "response", "answer"]:
            val = output.get(key)
            if val and isinstance(val, str):
                return val
                
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
                    
        str_vals = [v for v in output.values() if isinstance(v, str)]
        if len(str_vals) == 1:
            return str_vals[0]
            
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
                    content = chunk.content if hasattr(chunk, "content") else chunk.get("content", "") if isinstance(chunk, dict) else str(chunk)
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

# --- API Endpoint Router ---

@router.post("/chat")
async def chat(inputLangchain: InputLangchain):
    # Clean the input message
    clearMessage = inputLangchain.userMessage.split("\n\n###")[0].strip()
    
    # Initialize the LLM (Gemini 2.5 Flash Lite is very fast and supports tool calling)
    # model = ChatGoogleGenerativeAI(
    #     model="gemini-2.5-flash-lite",
    #     temperature=0.1,
    #     api_key=GEMINI_KEY
    # )
    model = ChatOpenAI(
        model="openrouter/free",
        api_key=OPENROUTER_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1
    )
    
    # Define custom tools extracted from MCP logic
    tools = [list_tables, get_table_schema, get_table_schema_with_relations, execute_sql]
    
    # Custom system prompt guiding the Agentic Loop
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""Bạn là một chuyên gia phân tích dữ liệu chuyên nghiệp và thân thiện.
Nhiệm vụ của bạn là hỗ trợ người dùng truy vấn và trả lời các câu hỏi về cơ sở dữ liệu MariaDB (database hiện tại là: '{ZOHO_MYSQL_DB}') bằng cách sử dụng các công cụ SQL an toàn được cung cấp.

Quy trình làm việc chuẩn của bạn:
1. Nếu chưa biết cấu trúc database hoặc bảng nào chứa thông tin cần tìm, hãy sử dụng `list_tables` để xem danh sách bảng.
2. Dùng `get_table_schema` hoặc `get_table_schema_with_relations` để tìm hiểu cấu trúc chi tiết của các bảng liên quan đến câu hỏi (tên cột, kiểu dữ liệu, các liên kết khóa ngoại). KHÔNG tự đoán cấu trúc bảng.
3. Tạo câu lệnh SQL SELECT chính xác để lấy thông tin cần thiết.
4. Chạy câu lệnh SQL bằng công cụ `execute_sql`.
5. Nếu câu lệnh SQL bị lỗi cú pháp hoặc lỗi DB, hãy phân tích thông báo lỗi, tự động sửa lỗi và thử lại bằng cách chạy lại `execute_sql`.
6. Diễn giải kết quả nhận được bằng tiếng Việt một cách rõ ràng và trực quan cho người dùng. Định dạng kết quả dạng bảng Markdown để dễ theo dõi nếu phù hợp.

Hướng dẫn định hướng nghiệp vụ (Mapping Nghiệp vụ - Bảng):
- Nghiệp vụ Hoàn ứng: Dùng bảng `clearance_payment` và các bảng liên quan (`clearance_payment_info`, `clearance_payment_attachment`).
- Nghiệp vụ Bảo lãnh: Dùng bảng `gruantee` và các bảng liên quan (`gruantee_attachment`) (Lưu ý: Tên bảng viết sai chính tả là 'gruantee', không phải 'guarantee').
- Nghiệp vụ Yêu cầu/Đề nghị thanh toán: Dùng bảng `request_payment` và các bảng liên quan (`request_payment_info`, `request_payment_attachment`).
- Nghiệp vụ Mua sắm: Dùng bảng `purchase` và các bảng liên quan (`purchase_info`, `purchase_detail`, `purchase_amount`, `purchase_attachment`).
- Nghiệp vụ Đấu thầu: Dùng bảng `bidding` và các bảng liên quan (`bidding_detail`, `bidding_assign`).
- Nghiệp vụ Nhân sự, Lương & Hợp đồng lao động: Dùng các bảng `employees`, `employees_salary`, `employees_subsidy`, `contract_human`, `payslip`, `payslip_bonus`, `payslip_deduction`, `department`, `leave`, `holiday`.
- Nghiệp vụ Hợp đồng dự án: Dùng bảng `contract_project` và các bảng liên quan (`contract_project_detail`, `contract_project_payment`, `contract_project_follow`, `contract_project_incident`, `contract_project_insurance`).
- Nghiệp vụ Hóa đơn xuất khẩu: Dùng bảng `export_invoice` và các bảng liên quan (`export_invoice_goods`).

Quy định bảo mật:
- Chỉ thực hiện các câu lệnh đọc dữ liệu (SELECT, SHOW, DESCRIBE...). Cấm chạy các lệnh làm thay đổi dữ liệu hoặc cấu trúc DB.
- Trả lời thân thiện và lịch sự bằng tiếng Việt."""),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create the agent executor using Google's tool calling capability
    agent = create_tool_calling_agent(model, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        max_iterations=10
    )
    
    # Bind variables to prompt and return streaming response
    return StreamingResponse(
        event_generator(
            agent_executor, 
            input_data={"input": clearMessage}
        ),
        media_type="text/event-stream"
    )
