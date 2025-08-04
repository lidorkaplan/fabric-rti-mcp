import inspect
import os
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from azure.kusto.data import (
    ClientRequestProperties,
    KustoConnectionStringBuilder,
)

from fabric_rti_mcp import __version__  # type: ignore
from fabric_rti_mcp.kusto.kusto_connection import KustoConnection
from fabric_rti_mcp.kusto.kusto_response_formatter import format_results


class KustoConnectionWrapper(KustoConnection):
    def __init__(
        self, cluster_uri: str, default_database: str, description: Optional[str] = None
    ):
        super().__init__(cluster_uri)
        self.default_database = default_database
        self.description = description or cluster_uri


class KustoConnectionCache(defaultdict[str, KustoConnectionWrapper]):
    def __init__(self) -> None:
        super().__init__()
        default_cluster = os.getenv("KUSTO_SERVICE_URI")
        default_db = os.getenv(
            "KUSTO_SERVICE_DEFAULT_DB",
            KustoConnectionStringBuilder.DEFAULT_DATABASE_NAME,
        )
        if default_cluster:
            self.add_cluster_internal(default_cluster, default_db, "default cluster")

    def __missing__(self, key: str) -> KustoConnectionWrapper:
        client = KustoConnectionWrapper(key, DEFAULT_DB)
        self[key] = client
        return client

    def add_cluster_internal(
        self,
        cluster_uri: str,
        default_database: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """Internal method to add a cluster during cache initialization."""
        cluster_uri = cluster_uri.strip()
        if cluster_uri.endswith("/"):
            cluster_uri = cluster_uri[:-1]

        if cluster_uri in self:
            return
        self[cluster_uri] = KustoConnectionWrapper(
            cluster_uri, default_database or DEFAULT_DB, description
        )


def add_kusto_cluster(
    cluster_uri: str,
    default_database: Optional[str] = None,
    description: Optional[str] = None,
) -> None:
    KUSTO_CONNECTION_CACHE.add_cluster_internal(
        cluster_uri, default_database, description
    )


def get_kusto_connection(cluster_uri: str) -> KustoConnectionWrapper:
    # clean uo the cluster URI since agents can send messy inputs
    cluster_uri = cluster_uri.strip()
    if cluster_uri.endswith("/"):
        cluster_uri = cluster_uri[:-1]
    return KUSTO_CONNECTION_CACHE[cluster_uri]


def _execute(
    query: str,
    cluster_uri: str,
    readonly_override: bool = False,
    database: Optional[str] = None,
) -> List[Dict[str, Any]]:
    caller_frame = inspect.currentframe().f_back  # type: ignore
    # Get the name of the caller function
    action = caller_frame.f_code.co_name  # type: ignore

    connection = get_kusto_connection(cluster_uri)
    client = connection.query_client

    # agents can send messy inputs
    query = query.strip()

    database = database or connection.default_database
    database = database.strip()

    crp: ClientRequestProperties = ClientRequestProperties()
    crp.application = f"fabric-rti-mcp{{{__version__}}}"  # type: ignore
    crp.client_request_id = f"KFRTI_MCP.{action}:{str(uuid.uuid4())}"  # type: ignore
    if action not in DESTRUCTIVE_TOOLS and not readonly_override:
        crp.set_option("request_readonly", True)
    result_set = client.execute(database, query, crp)
    return format_results(result_set)


def kusto_get_clusters() -> List[Tuple[str, str]]:
    """
    Retrieves a list of all Kusto clusters in the cache.

    :return: List of tuples containing cluster URI and description. When selecting a cluster,
             the URI must be used, the description is used only for additional information.
    """
    return [(uri, client.description) for uri, client in KUSTO_CONNECTION_CACHE.items()]


def kusto_connect(
    cluster_uri: str, default_database: str, description: Optional[str] = None
) -> None:
    """
    Connects to a Kusto cluster and adds it to the cache.

    :param cluster_uri: The URI of the Kusto cluster.
    :param default_database: The default database to use for queries on this cluster.
    :param description: Optional description for the cluster. Cannot be used to retrieve the cluster,
                       but can be used to provide additional information about the cluster.
    """
    add_kusto_cluster(
        cluster_uri, default_database=default_database, description=description
    )


def kusto_query(
    query: str, cluster_uri: str, database: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Executes a KQL query on the specified database. If no database is provided,
    it will use the default database.

    ## Fallback Strategy
    When struggling to generate a query, consider using the `kusto_get_shots` tool to retrieve 
    existing high quality shots that can provide context or examples for query structure or logic.
    When struggling to generate a query, consider creating new high quality shots using the
    workflow: select shots (check UDF availability for shot selection functions) → use explain results tool → embed and ingest.
    New shots can improve the quality of future queries.

    :param query: The KQL query to execute.
    :param cluster_uri: The URI of the Kusto cluster.
    :param database: Optional database name. If not provided, uses the default database.
    :return: The result of the query execution as a list of dictionaries (json).
    """
    return _execute(query, cluster_uri, database=database)


def kusto_command(
    command: str, cluster_uri: str, database: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Executes a kusto management command on the specified database. If no database is provided,
    it will use the default database.

    :param command: The kusto management command to execute.
    :param cluster_uri: The URI of the Kusto cluster.
    :param database: Optional database name. If not provided, uses the default database.
    :return: The result of the command execution as a list of dictionaries (json).
    """
    return _execute(command, cluster_uri, database=database)


def kusto_list_databases(cluster_uri: str) -> List[Dict[str, Any]]:
    """
    Retrieves a list of all databases in the Kusto cluster.

    :param cluster_uri: The URI of the Kusto cluster.
    :return: List of dictionaries containing database information.
    """
    return _execute(".show databases", cluster_uri)


def kusto_list_tables(cluster_uri: str, database: str) -> List[Dict[str, Any]]:
    """
    Retrieves a list of all tables in the specified database.

    :param cluster_uri: The URI of the Kusto cluster.
    :param database: The name of the database to list tables from.
    :return: List of dictionaries containing table information.
    """
    return _execute(".show tables", cluster_uri, database=database)


def kusto_get_entities_schema(
    cluster_uri: str, database: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves schema information for all entities (tables, materialized views, functions)
    in the specified database. If no database is provided, uses the default database.

    :param cluster_uri: The URI of the Kusto cluster.
    :param database: Optional database name. If not provided, uses the default database.
    :return: List of dictionaries containing entity schema information.
    """
    return _execute(
        ".show databases entities with (showObfuscatedStrings=true) "
        f"| where DatabaseName == '{database or DEFAULT_DB}' "
        "| project EntityName, EntityType, Folder, DocString",
        cluster_uri,
        database=database,
    )


def kusto_get_table_schema(
    table_name: str, cluster_uri: str, database: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves the schema information for a specific table in the specified database.
    If no database is provided, uses the default database.

    :param table_name: Name of the table to get schema for.
    :param cluster_uri: The URI of the Kusto cluster.
    :param database: Optional database name. If not provided, uses the default database.
    :return: List of dictionaries containing table schema information.
    """
    return _execute(
        f".show table {table_name} cslschema", cluster_uri, database=database
    )


def kusto_get_function_schema(
    function_name: str, cluster_uri: str, database: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves schema information for a specific function, including parameters and output schema.
    If no database is provided, uses the default database.

    :param function_name: Name of the function to get schema for.
    :param cluster_uri: The URI of the Kusto cluster.
    :param database: Optional database name. If not provided, uses the default database.
    :return: List of dictionaries containing function schema information.
    """
    return _execute(f".show function {function_name}", cluster_uri, database=database)


def kusto_sample_table_data(
    table_name: str,
    cluster_uri: str,
    sample_size: int = 10,
    database: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves a random sample of records from the specified table.
    If no database is provided, uses the default database.

    :param table_name: Name of the table to sample data from.
    :param cluster_uri: The URI of the Kusto cluster.
    :param sample_size: Number of records to sample. Defaults to 10.
    :param database: Optional database name. If not provided, uses the default database.
    :return: List of dictionaries containing sampled records.
    """
    return _execute(
        f"{table_name} | sample {sample_size}", cluster_uri, database=database
    )


def kusto_sample_function_data(
    function_call_with_params: str,
    cluster_uri: str,
    sample_size: int = 10,
    database: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves a random sample of records from the result of a function call.
    If no database is provided, uses the default database.

    :param function_call_with_params: Function call string with parameters.
    :param cluster_uri: The URI of the Kusto cluster.
    :param sample_size: Number of records to sample. Defaults to 10.
    :param database: Optional database name. If not provided, uses the default database.
    :return: List of dictionaries containing sampled records.
    """
    return _execute(
        f"{function_call_with_params} | sample {sample_size}",
        cluster_uri,
        database=database,
    )


def kusto_ingest_inline_into_table(
    table_name: str,
    data_comma_separator: str,
    cluster_uri: str,
    database: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Ingests inline CSV data into a specified table. The data should be provided as a comma-separated string.
    If no database is provided, uses the default database.

    :param table_name: Name of the table to ingest data into.
    :param data_comma_separator: Comma-separated data string to ingest.
    :param cluster_uri: The URI of the Kusto cluster.
    :param database: Optional database name. If not provided, uses the default database.
    :return: List of dictionaries containing the ingestion result.
    """
    return _execute(
        f".ingest inline into table {table_name} <| {data_comma_separator}",
        cluster_uri,
        database=database,
    )


def kusto_get_shots(prompt: str,
    shots_table_name: str,
    cluster_uri: str,
    sample_size: int = 3,
    database: Optional[str] = None,
    embedding_endpoint: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves shots that are most semantic similar to the supplied prompt from the specified shots table.
    This tool can also help retrieve similar shots when the copilot fails at generating the right query 
    for any reason. Calling this tool can bring similar shots and help with kql generation attempts.

    ## Shot Retrieval Guidelines
    Consider using this tool to retrieve similar high quality shots to provide context or examples 
    for query structure or logic. Use shots to understand patterns and improve query generation quality.
    This is particularly useful as a fallback strategy when struggling to generate queries.

    :param prompt: The user prompt to find similar shots for.
    :param shots_table_name: Name of the table containing the shots. The table should have "EmbeddingText" (string) column
                             containing the natural language prompt, "AugmentedText" (string) column containing the respective KQL,
                             and "EmbeddingVector" (dynamic) column containing the embedding vector for the NL.
    :param cluster_uri: The URI of the Kusto cluster.
    :param sample_size: Number of most similar shots to retrieve. Defaults to 3.
    :param database: Optional database name. If not provided, uses the "AI" database or the default database.
    :param embedding_endpoint: Optional endpoint for the embedding model to use. 
                             If not provided, uses the AZ_OPENAI_EMBEDDING_ENDPOINT environment variable.
                             If no valid endpoint is set, this function should not be called.
    :return: List of dictionaries containing the shots records.
    """
    
    # Use provided endpoint, or fall back to environment variable, or use default
    endpoint = embedding_endpoint or DEFAULT_EMBEDDING_ENDPOINT

    kql_query = f"""
        let model_endpoint = '{endpoint}';
        let embedded_term = toscalar(evaluate ai_embeddings('{prompt}', model_endpoint));
        {shots_table_name}
        | extend similarity = series_cosine_similarity(embedded_term, EmbeddingVector)
        | top {sample_size} by similarity
        | project similarity, EmbeddingText, AugmentedText
    """

    return _execute(kql_query, cluster_uri, database=database)


def kusto_embed_and_ingest_shots( # ingest_and_embed_shots
    cluster_uri: str,
    input_data: List[Dict[str, Any]],
    shots_table_name: str = "KustoCopilotEmbeddings",
    database: Optional[str] = None,
    embedding_endpoint: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Creates embeddings from input data and ingests them into the shots table.
    Validates that the shots table exists with the correct schema and that input data has required columns.
    Filters out duplicate entries based on the hash of AugmentedText.

    ## Creating and Storing New Shots Workflow
    **Do not create/generate new shots on your own** - always use shots from user created queries.
    This tool is part of the workflow for getting new high quality shots:
    1. **Select shots**: check UDF availability for shot selection functions
    2. **Use kusto_explain_kql_results tool** to get natural language descriptions
    3. **Embed and ingest** using this tool
    
    New shots can improve the quality of future queries. Follow the workflow outlined above when creating new shots.

    :param cluster_uri: The URI of the Kusto cluster.
    :param input_data: List of dictionaries containing the data to process. Must contain:
                       - 'EmbeddingText': Natural language description of a shot/example query
                       - 'AugmentedText': KQL query shot/example corresponding to the description
                       Any additional columns that exist in the shots table schema should also be included.
    :param shots_table_name: Name of the table to ingest embeddings into. Defaults to "KustoCopilotEmbeddings".
                             The table must have the following required columns:
                             - Timestamp (datetime): When the record was created
                             - Key (string): Unique identifier for the record (hash of AugmentedText)
                             - EmbeddingText (string): Natural language text that was embedded
                             - AugmentedText (string): The corresponding KQL query or response
                             - EmbeddingVector (dynamic): The embedding vector for the EmbeddingText
                             - EmbeddingModel (string): The model used to generate the embedding
                             - Metadata (dynamic): Additional metadata for the record
                             - User (string): The user who created the record
    :param database: Optional database name. If not provided, uses the default database.
    :param embedding_endpoint: Optional endpoint for the embedding model to use. 
                              If not provided, uses the AZ_OPENAI_EMBEDDING_ENDPOINT environment variable.
    :return: List of dictionaries containing the ingestion result.
    """
    # Use provided endpoint, or fall back to environment variable, or use default
    endpoint = embedding_endpoint or DEFAULT_EMBEDDING_ENDPOINT
    
    if not endpoint:
        raise ValueError("Embedding endpoint is required. Set AZ_OPENAI_EMBEDDING_ENDPOINT environment variable or provide embedding_endpoint parameter.")
    
    # Validate input data is not empty
    if not input_data:
        raise ValueError("Input data cannot be empty.")
    
    # Step 1: Validate shots table exists and has correct schema
    try:
        schema_result = kusto_get_table_schema(shots_table_name, cluster_uri, database)
        if not schema_result:
            raise ValueError(f"Table '{shots_table_name}' does not exist.")
        
        # Define required columns with their expected types
        shots_table_required_columns = {
            "Timestamp": "datetime",
            "Key": "string", 
            "EmbeddingText": "string",
            "AugmentedText": "string",
            "EmbeddingVector": "dynamic",
            "EmbeddingModel": "string",
            "Metadata": "dynamic",
            "User": "string"
        }
        
        # Parse schema to get column info
        shots_table_schema: Dict[str, str] = {}
        schema_text = schema_result[0]['Schema']
        
        # Parse the schema string to extract column definitions
        for col in schema_text.split(','):
            parts = col.split(':')
            if len(parts) >= 2:
                col_name = parts[0].strip()
                col_type = parts[1].strip()
                shots_table_schema[col_name] = col_type
        
        # Validate required columns exist with correct types
        missing_columns: List[str] = []
        type_mismatches: List[str] = []
        optional_columns: List[str] = []
        
        for col_name, expected_type in shots_table_required_columns.items():
            if col_name not in shots_table_schema:
                missing_columns.append(col_name)
            elif shots_table_schema[col_name] != expected_type:
                type_mismatches.append(f"Column '{col_name}' has type '{shots_table_schema[col_name]}' but expected '{expected_type}'")

        for col_name in shots_table_schema.keys():
            if col_name not in shots_table_required_columns:
                optional_columns.append(col_name)
        
        error_message: str = ""
        if missing_columns:
            error_message += f"missing required columns: {', '.join(missing_columns)}\n"
        if type_mismatches:
            error_message += f"has incorrect column types: {chr(10).join(type_mismatches)}\n"

        if error_message:
            raise ValueError(f"Table '{shots_table_name}' {error_message.strip()}")
                
    except Exception as e:
        raise ValueError(f"Failed to validate shots table schema: {str(e)}")
    
    # Step 2: Validate input data has required columns and compatible types
    try:
        required_input_columns = ["EmbeddingText", "AugmentedText"] + optional_columns
        optional_columns += ["Metadata", "User"]
        first_row = input_data[0]
        
        # Check for missing required columns in input data
        missing_input_columns: List[str] = []
        for col in required_input_columns:
            if col not in first_row:
                missing_input_columns.append(col)
        
        extra_columns: List[str] = []
        # Check for extra columns in input data
        for col in first_row.keys():
            if col not in required_input_columns and col not in optional_columns:
                extra_columns.append(col)
                
        error_message: str = ""
        if missing_input_columns:
            error_message += f"Input data is missing required columns: {', '.join(missing_input_columns)} \n"
        if extra_columns:
            error_message += f"Input data contains unexpected columns: {', '.join(extra_columns)} \n"
            
        if error_message:
            raise ValueError(error_message.strip())
        
    except Exception as e:
        raise ValueError(f"Failed to validate input data: {str(e)}")    
        
    
    # Step 3: Prepare data for KQL datatable
    data_header: List[str] = [f"{col}:{shots_table_schema[col]}" for col in input_data[0].keys()]
    # convert input data to comma-separated string for all input rows and all columns
    data_rows: List[str] = []
    for row in input_data:
        row_string = '```,```'.join(str(value) for value in row.values())
        row_string = f'```{row_string}```'  # wrap in quotes to handle commas
        data_rows.append(row_string)
    data_table_rows = ",\n    ".join(data_rows)
    
    # Extract embedding model from endpoint between deployments/ and ;
    embedding_model = endpoint.split("deployments/")[-1].split(";")[0]
    
    # Step 4: Create KQL query to process embeddings and ingest
    # Filters out existing embeddings based on the hash of AugmentedText
    # Will require different handling if we want to support updating embedded text
    # TODO: get user
    ingest_query = f"""
    let model_endpoint = '{endpoint}';
    let input_data = datatable({','.join(data_header)})
    [
        {data_table_rows}
    ];
    {shots_table_name}
    | take 0
    | union
    (input_data
    | extend Timestamp = now()
    | extend Key = hash_sha256(AugmentedText)
    | join kind = leftanti {shots_table_name} on Key
    | evaluate ai_embeddings(EmbeddingText, model_endpoint)
    | project-rename EmbeddingVector = EmbeddingText_embeddings
    | extend EmbeddingModel = '{embedding_model}'
    )
    """
    
    # Execute the ingest command
    insert_command = f".set-or-append {shots_table_name} <| {ingest_query}"
    ingestion_result = _execute(insert_command, cluster_uri, database=database)
    return ingestion_result
    
def kusto_explain_kql_results(
    kql_query: str,
    cluster_uri: str,
    database: Optional[str] = None,
    completion_endpoint: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Use this tool to explain KQL queries and their results in natural language. When users ask to "explain queries", 
    "explain results", "what do these queries do", or want natural language descriptions of query data, 
    this tool modifies the provided KQL query to add AI-generated explanations for each result row and executes it.
    
    Perfect for: explaining query history, describing what queries do, generating human-readable summaries of data.
    The tool uses AI completion to convert technical query results into easy-to-understand explanations.

    ## Creating New Shots Workflow
    This tool is a key component in the workflow for getting new high quality shots:
    1. **Select shots**: check UDF availability for shot selection functions  
    2. **Use this tool** to get natural language descriptions (Input: KQL query → Output: NL description appended to each result)
    3. **Embed and ingest** using the kusto_embed_and_ingest_shots tool
    
    The natural language descriptions generated by this tool can be used as EmbeddingText for creating new shots.
    
    :param kql_query: a '.show queries' query or any other KQL query that returns results. 
                        A natural language description will be generated for each row in the results.
    :param cluster_uri: The URI of the Kusto cluster.
    :param database: Optional database name. If not provided, uses the default database.
    :param completion_endpoint: Optional endpoint for the text completion model to use.
                               If not provided, uses the AZ_OPENAI_COMPLETION_ENDPOINT environment variable.
                               If no valid endpoint is set, this function should not be called.
    :return: The result of the query execution with natural language descriptions as a list of dictionaries (json).
    """

    if not kql_query.strip():
        raise ValueError("KQL query cannot be empty")

    # Use provided endpoint, or fall back to environment variable, or use default
    endpoint = completion_endpoint or DEFAULT_COMPLETION_ENDPOINT

    if not endpoint:
        raise ValueError("No completion endpoint provided. Set AZ_OPENAI_COMPLETION_ENDPOINT environment variable or provide completion_endpoint parameter.")

    # Clean up the input query
    cleaned_query = kql_query.strip()

    # Create a working query that calls AI to explain each row using ai_chat_completion_prompt
    # Avoid let statement to prevent syntax issues with complex queries
    modified_query = f"""{cleaned_query}
| extend explanation_prompt = strcat("Provide a natural language request that would result in this KQL query, return the NL description only. No other words or examples: ", tostring(pack_all()))
| evaluate ai_chat_completion_prompt(explanation_prompt, '{endpoint}')
| extend NaturalLanguageDescription = explanation_prompt_chat_completion
| project-away explanation_prompt, explanation_prompt_chat_completion"""

    # Execute the modified query and return the results
    return _execute(modified_query, cluster_uri, database=database)

KUSTO_CONNECTION_CACHE: KustoConnectionCache = KustoConnectionCache()
DEFAULT_DB = KustoConnectionStringBuilder.DEFAULT_DATABASE_NAME
DEFAULT_EMBEDDING_ENDPOINT = os.getenv("AZ_OPENAI_EMBEDDING_ENDPOINT")
DEFAULT_COMPLETION_ENDPOINT = os.getenv("AZ_OPENAI_COMPLETION_ENDPOINT")

DESTRUCTIVE_TOOLS = {
    kusto_command.__name__,
    kusto_ingest_inline_into_table.__name__,
    kusto_embed_and_ingest_shots.__name__,
}
