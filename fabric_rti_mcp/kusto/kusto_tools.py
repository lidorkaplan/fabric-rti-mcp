from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from fabric_rti_mcp.kusto import kusto_service
from fabric_rti_mcp.kusto.renderers import Renderers


def register_tools(mcp: FastMCP) -> None:
    mcp.add_tool(
        kusto_service.kusto_get_clusters,
        annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
    )
    mcp.add_tool(
        kusto_service.add_kusto_cluster,
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
    )
    mcp.add_tool(
        kusto_service.kusto_query,
        annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
    )
    mcp.add_tool(
        kusto_service.kusto_command,
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True),
    )
    mcp.add_tool(
        kusto_service.kusto_list_databases,
        annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
    )
    mcp.add_tool(
        kusto_service.kusto_list_tables,
        annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
    )
    mcp.add_tool(
        kusto_service.kusto_get_entities_schema,
        annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
    )
    mcp.add_tool(
        kusto_service.kusto_get_table_schema,
        annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
    )
    mcp.add_tool(
        kusto_service.kusto_get_function_schema,
        annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
    )
    mcp.add_tool(
        kusto_service.kusto_sample_table_data,
        annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
    )
    mcp.add_tool(
        kusto_service.kusto_sample_function_data,
        annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
    )
    mcp.add_tool(
        kusto_service.kusto_ingest_inline_into_table,
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
    )
    mcp.add_tool(
        kusto_service.kusto_get_shots,
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
    )
    mcp.add_tool(
        Renderers.render_piechart,
        description="Renders a pie chart with the specified data and options, based on the query result."
    )
    mcp.add_tool(
        Renderers.render_table,
        description="Renders a table with the specified data and options, based on the query result."
    )
    mcp.add_tool(
        Renderers.render_barchart,
        description="Renders a bar chart with the specified data and options. Supports DEFAULT (bar), STACKED (stackedbar), and STACKED100 (stacked100bar) chart types."
    )
    mcp.add_tool(
        Renderers.render_columnchart,
        description="Renders a column chart with the specified data and options. Supports DEFAULT (column), STACKED (stackedcolumn), and STACKED100 (stacked100column) chart types."
    )
    mcp.add_tool(
        Renderers.render_areachart,
        description="Renders an area chart with the specified data and options. Supports DEFAULT (area), STACKED (stackedarea), and STACKED100 (stacked100area) chart types."
    )
    mcp.add_tool(
        Renderers.render_linechart,
        description="Renders a line chart with the specified data and options."
    )
    mcp.add_tool(
        Renderers.render_card,
        description="Renders a card visualization with the specified data and options, based on the query result."
    )
    mcp.add_tool(
        Renderers.render_multistat,
        description="Renders a multi-stat visualization with the specified data and options, based on the query result."
    )
    mcp.add_tool(
        Renderers.render_scatterchart,
        description="Renders a scatter chart with the specified data and options."
    )
    mcp.add_tool(
        Renderers.render_funnelchart,
        description="Renders a funnel chart with the specified data and options."
    )
    mcp.add_tool(
        Renderers.render_mapchart,
        description="Renders a map visualization with the specified data and options."
    )
    mcp.add_tool(
        Renderers.render_timechart,
        description="Renders a time chart with the specified data and options. The x-axis must be a datetime column."
    )
    mcp.add_tool(
        Renderers.render_anomalychart,
        description="Renders an anomaly chart with the specified data and options."
    )
    mcp.add_tool(
        Renderers.render_heatmap,
        description="Renders a heatmap visualization with the specified data and options."
    )
