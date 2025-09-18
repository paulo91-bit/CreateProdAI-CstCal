import logging
import azure.functions as func
import os
import json
from datetime import datetime, timezone
import requests
import tempfile
from openai import OpenAI
import docx2txt
from PyPDF2 import PdfReader

# Imports for existing "CalculateAndReportCosts" function
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import base64
from azure.storage.blob import BlobServiceClient
import psycopg2
from psycopg2.extras import RealDictCursor

# --- Initialize the Function App ---
# Each function will define its own auth_level in the @app.route decorator
app = func.FunctionApp()

# --- HELPER FUNCTIONS ---

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    host = os.getenv("PGHOST")
    port = os.getenv("PGPORT", "5432")
    dbname = os.getenv("PGDATABASE")
    user = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")
    sslmode = "require"
    conn = psycopg2.connect(
        host=host, port=port, dbname=dbname, user=user, password=password, sslmode=sslmode
    )
    return conn

def clean_value(v):
    """Converts empty strings to None so Postgres can accept NULL values."""
    if v is None:
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    return v


def insert_usage_tracking(data: dict):
    """Inserts a detailed usage row into the public.usage_tracking table."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Ensure IDs default to 0 if missing/empty
        if not data.get("location_id"):
            data["location_id"] = 0
        if not data.get("company_id"):
            data["company_id"] = 0

        # Clean payload before inserting
        cleaned_data = {k: clean_value(v) for k, v in data.items()}

        logging.debug(f"🧹 Cleaned payload going into Postgres: {json.dumps(cleaned_data, indent=2, default=str)}")

        insert_query = """
        INSERT INTO public.usage_tracking (
            location_id, provider, model, workflow_name, workflow_url,
            input_tokens, input_tokens_cached, input_tokens_uncached,
            output_tokens, output_tokens_cached, output_tokens_uncached,
            buildship_execution_time, buildship_node_qty,
            cost_usd, buildship_credits_used, notes
        ) VALUES (
            %(location_id)s, %(provider)s, %(model)s, %(workflow_name)s, %(workflow_url)s,
            %(input_tokens)s, %(input_tokens_cached)s, %(input_tokens_uncached)s,
            %(output_tokens)s, %(output_tokens_cached)s, %(output_tokens_uncached)s,
            %(buildship_execution_time)s, %(buildship_node_qty)s,
            %(cost_usd)s, %(buildship_credits_used)s, %(notes)s
        );
        """
        cur.execute(insert_query, cleaned_data)
        conn.commit()
        cur.close()
        conn.close()
        logging.info("✅ usage_tracking row inserted successfully.")
    except Exception as e:
        logging.error(f"❌ Failed to insert usage_tracking row: {e}", exc_info=True)



def extract_text(file) -> str:
    """Extracts text content from .txt, .pdf, and .docx files."""
    filename = file.filename.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    text = ""
    try:
        if filename.endswith(".txt"):
            with open(tmp_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif filename.endswith(".pdf"):
            reader = PdfReader(tmp_path)
            text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif filename.endswith(".docx"):
            text = docx2txt.process(tmp_path)
        else:
            raise ValueError(f"Unsupported file type: {filename}")
    finally:
        os.remove(tmp_path)
    return text

# --- AZURE FUNCTIONS ---

@app.function_name(name="GenerateProduct")
@app.route(route="generate-product", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
def generate_product(req: func.HttpRequest) -> func.HttpResponse:
    """
    Generates product details from a prompt and an optional file using OpenAI.
    Accepts multipart/form-data.
    """
    logging.info("--- GenerateProduct function started. ---")
    try:
        prompt = req.form.get("prompt")
        api_key = req.form.get("apiKey")
        location_id = req.form.get("locationId")
        company_id = req.form.get("companyId")
        workflow_name = req.form.get("workflow_name", "product_generation_workflow")
        model_name = "gpt-4o-mini"
        file = req.files.get("file")

        if not all([prompt, api_key, location_id, company_id]):
            logging.error("Validation failed: Missing required form fields.")
            return func.HttpResponse(
                "Request form must include 'prompt', 'apiKey', 'locationId', and 'companyId'.",
                status_code=400
            )

        document_context = ""
        if file:
            logging.info(f"Extracting text from uploaded file: {file.filename}")
            document_context = extract_text(file)
            logging.info(f"Extracted {len(document_context)} characters from document.")
        
        combined_input = (
            f"Please use the following document as context:\n---CONTEXT---\n{document_context}\n---END CONTEXT---\n\n"
            f"Now, based on that context, please follow this instruction:\n---PROMPT---\n{prompt}\n---END PROMPT---"
        )
        
        logging.info("Sending request to OpenAI...")
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI that creates product details based on provided context and a user prompt. "
                        "Return a JSON object with keys: productName, productDescription, productCategory."
                    )
                },
                {"role": "user", "content": combined_input}
            ],
            temperature=0.7,
            max_tokens=800
        )

        result_content = response.choices[0].message.content
        token_usage = response.usage
        logging.info(f"Received response from OpenAI. Token usage: {token_usage}")

        try:
            result_json = json.loads(result_content)
        except json.JSONDecodeError:
            logging.warning("Failed to parse JSON directly, attempting to extract from string.")
            start = result_content.find("{")
            end = result_content.rfind("}") + 1
            if start != -1 and end != 0:
                result_json = json.loads(result_content[start:end])
            else:
                raise ValueError("Failed to parse JSON from OpenAI response.")

        output = {
            "analysis": result_json,
            "location_id": location_id,
            "company_id": company_id,
            "workflow_name": workflow_name,
            "token_usage": {
                "provider": "openai",
                "model": model_name,
                "input_tokens": token_usage.prompt_tokens,
                "output_tokens": token_usage.completion_tokens,
            }
        }

        # --- NEW: invoke Logic App endpoint with product + token usage payload ---
        try:
            logic_app_url = "https://prod-02.westus2.logic.azure.com:443/workflows/17d39412c2924ea78881b1cae2bdb6ba/triggers/When_an_HTTP_request_is_received/paths/invoke?api-version=2016-10-01&sp=%2Ftriggers%2FWhen_an_HTTP_request_is_received%2Frun&sv=1.0&sig=XoMM_Yz75hzUBOVxutbxm08Ja9fPyCdHMwSVpEPWafE"
            logic_payload = {
                "productName": result_json.get("productName"),
                "productDescription": result_json.get("productDescription"),
                "productCategory": result_json.get("productCategory"),
                "location_id": location_id,
                "company_id": company_id,
                "workflow_name": workflow_name,
                "token_usage": {
                    "provider": "openai",
                    "model": model_name,
                    "input_tokens": token_usage.prompt_tokens,
                    "output_tokens": token_usage.completion_tokens
                }
            }
            resp = requests.post(logic_app_url, json=logic_payload, timeout=10)
            resp.raise_for_status()
            logging.info(f"✅ Logic App invoked successfully: {resp.status_code}")
        except Exception as e:
            logging.error(f"❌ Failed to invoke Logic App: {e}", exc_info=True)
        # --- end Logic App invocation ---

        return func.HttpResponse(
            json.dumps(output), mimetype="application/json", status_code=200
        )

    except Exception as e:
        logging.error(f"Error in GenerateProduct: {e}", exc_info=True)
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)

@app.function_name(name="GetProviderCosts")
@app.route(route="get-provider-costs", methods=["GET"], auth_level=func.AuthLevel.FUNCTION)
def get_provider_costs(req: func.HttpRequest) -> func.HttpResponse:
    """Reads and returns the provider_costs.json file from Azure Blob Storage."""
    logging.info('GetProviderCosts function processed a request.')
    try:
        connect_str = os.environ.get("COST_DATA_CONNECTION_STRING")
        if not connect_str:
            raise ValueError("COST_DATA_CONNECTION_STRING environment variable not set.")
        container_name = "configs"
        blob_name = "provider_costs.json"
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        downloader = blob_client.download_blob(max_connections=1)
        provider_costs = json.loads(downloader.readall())
        return func.HttpResponse(json.dumps(provider_costs), mimetype="application/json", status_code=200)
    except Exception as e:
        logging.error(f"Error fetching provider costs from blob storage: {e}", exc_info=True)
        return func.HttpResponse(f"Error fetching provider costs: {str(e)}", status_code=500)

@app.function_name(name="CalculateAndReportCosts")
@app.route(route="calculate-and-report-costs", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
def calculate_and_report_costs(req: func.HttpRequest) -> func.HttpResponse:
    """
    Calculates cost based on usage, logs it to a DB, and reports to an external API.
    Fetches pricing info from Blob Storage.
    """
    logging.info('--- CalculateAndReportCosts function received a request. ---')
    try:
        req_body = req.get_json()
    except ValueError:
        return func.HttpResponse("Invalid JSON in request body.", status_code=400)

    workflow_data = req_body.get('workflow_data')
    encryption_key = os.environ.get("MAISY3G_ENCRYPTION_KEY")
    encryption_iv = os.environ.get("MAISY3G_ENCRYPTION_IV")

    if not workflow_data:
        return func.HttpResponse("Missing 'workflow_data'.", status_code=400)
    if not encryption_key or not encryption_iv:
        return func.HttpResponse("Server encryption is not configured.", status_code=500)

    try:
        connect_str = os.environ.get("COST_DATA_CONNECTION_STRING")
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        blob_client = blob_service_client.get_blob_client(container="configs", blob="provider_costs.json")
        downloader = blob_client.download_blob()
        provider_costs = json.loads(downloader.readall())
    except Exception as e:
        logging.error(f"❌ Failed to fetch provider costs: {e}", exc_info=True)
        return func.HttpResponse(f"Could not fetch provider costs: {str(e)}", status_code=500)

    if isinstance(workflow_data, str):
        workflow_data = json.loads(workflow_data)
    
    encryption_key_bytes = base64.b64decode(encryption_key)
    iv_bytes = base64.b64decode(encryption_iv)

    total_workflow_cost, total_input_tokens, total_output_tokens = 0.0, 0, 0
    usage_list = workflow_data.get('usage', [])

    for entry in usage_list:
        try:
            provider = str(entry.get('provider', '')).strip().lower()
            model = str(entry.get('model', '')).strip().lower()
            input_tokens = float(entry.get('input_tokens', 0))
            output_tokens = float(entry.get('output_tokens', 0))
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            matched_entry = next(
                (row for row in provider_costs
                    if str(row.get("provider", "")).strip().lower() == provider
                    and str(row.get("model", "")).strip().lower() == model),
                None
            )
            
            if matched_entry:
                input_cpm = float(matched_entry.get('input_token_cpm', 0))
                output_cpm = float(matched_entry.get('output_token_cpm', 0))
                input_cost = (input_tokens / 1000000) * input_cpm
                output_cost = (output_tokens / 1000000) * output_cpm
                total_workflow_cost += (input_cost + output_cost)
            else:
                logging.warning(f"No cost info for {provider}:{model}. Skipped.")
        except Exception as e:
            logging.error(f"Error calculating entry cost: {e}", exc_info=True)

    cleartext_payload = {
        "workflow_name": workflow_data.get("workflow_name"),
        "cost_usd": f"{total_workflow_cost:.10f}"
        
    }
    company_id = workflow_data.get("company_id")
    location_id = workflow_data.get("location_id")
    if company_id:
        cleartext_payload["company_id"] = int(company_id)
    if location_id:
        cleartext_payload["location_id"] = int(location_id)

    insert_data = {
        "location_id": location_id or 0,
        "provider": usage_list[0].get("provider") if usage_list else "unknown",
        "model": usage_list[0].get("model") if usage_list else None,
        "workflow_name": workflow_data.get("workflow_name"),
        "workflow_url": workflow_data.get("workflow_url"),
        "input_tokens": total_input_tokens, "output_tokens": total_output_tokens,
        "input_tokens_cached": 0, "input_tokens_uncached": 0,
        "output_tokens_cached": 0, "output_tokens_uncached": 0,
        "buildship_execution_time": workflow_data.get("buildship_execution_time"),
        "buildship_node_qty": workflow_data.get("buildship_node_qty"),
        "cost_usd": f"{total_workflow_cost:.10f}",
        "buildship_credits_used": workflow_data.get("buildship_credits_used"),
        "notes": workflow_data.get("notes")
    }
    logging.info(f"💾 Data being sent to database: {json.dumps(insert_data, indent=2)}")
    insert_usage_tracking(insert_data)

    try:
        plaintext_payload_str = json.dumps(cleartext_payload)
        cipher = AES.new(encryption_key_bytes, AES.MODE_CBC, iv_bytes)
        padded_data = pad(plaintext_payload_str.encode("utf-8"), AES.block_size)
        encrypted_data_bytes = cipher.encrypt(padded_data)
        encrypted_payload_b64 = base64.b64encode(encrypted_data_bytes).decode("utf-8")
    except Exception as e:
        return func.HttpResponse(f"Encryption failed: {str(e)}", status_code=500)

    try:
        endpoint_url = "https://dev.maisy365.com/admin/creditUsed"
        headers = {"Content-Type": "application/json", "Signature": encrypted_payload_b64}
        response = requests.post(endpoint_url, headers=headers, json=cleartext_payload, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return func.HttpResponse(json.dumps({"status": "error", "message": f"API call failed: {str(e)}"}), status_code=500)

    return func.HttpResponse(json.dumps({"cost_details": cleartext_payload}), status_code=200, mimetype="application/json")

@app.function_name(name="GetUsageLogs")
@app.route(route="usage-logs", methods=["GET"], auth_level=func.AuthLevel.FUNCTION)
def get_usage_logs(req: func.HttpRequest) -> func.HttpResponse:
    """Fetches usage_tracking logs from Postgres."""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        limit = int(req.params.get("limit", 10))
        cur.execute("SELECT * FROM public.usage_tracking ORDER BY created_at DESC LIMIT %s;", (limit,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return func.HttpResponse(json.dumps(rows, default=str), mimetype="application/json", status_code=200)
    except Exception as e:
        logging.error(f"❌ Failed to fetch usage logs: {e}", exc_info=True)
        return func.HttpResponse(f"Error fetching logs: {str(e)}", status_code=500)
