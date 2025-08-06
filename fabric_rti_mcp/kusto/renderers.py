from typing import Any, Dict, List
from enum import Enum

class HeatmapColorPalette(Enum):
    BLUE = "blue"
    GREEN = "green"
    PURPLE = "purple"
    ORANGE = "orange"
    PINK = "pink"
    YELLOW = "yellow"

class DisplayOrientation(Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"

class ChartType(Enum):
    DEFAULT = ""
    STACKED = "stacked"
    STACKED100 = "stacked100"


class Renderers:
    @staticmethod
    def render_table(
        query_result: List[Dict[str, Any]],
        title: str,
        description: str,
    ) -> Dict[str, Any]:
        return {
            "visualType": "table",
            "title": title,
            "description": description,
            "visualOptions": {
                "table_enableRenderLinks": False,
                "colorRulesDisabled": False,
                "colorStyle": "light",
                "crossFilterDisabled": False,
                "drillthroughDisabled": False,
                "crossFilter": [],
                "drillthrough": [],
                "table__renderLinks": [],
                "colorRules": [] #TODO: consider adding color rules
            }
        }

    @staticmethod
    def render_piechart(
        query_result: List[Dict[str, Any]], 
        title: str, 
        description: str, 
        x_column: str = None, 
        y_columns: List[str] = None, 
        topNSlices: int = None
    ) -> Dict[str, Any]: #TODO: consider returning a VisualOptions object rather than undefined json (maybe not in Python POC)
        return {
            "visualType": "pie",
            "title": title,
            "description": description,
            "visualOptions": {
                "xColumn": x_column,
                "yColumns": y_columns,
                "pie__label": [
                    "name",
                    "percentage"
                ],
                "tooltipDisabled": False,
                "pie__tooltip": [
                    "name",
                    "percentage",
                    "value"
                ],
                "pie__orderBy": "size",
                "pie__kind": "pie",
                "pie__topNSlices": topNSlices,
                "legendLocation": "bottom",
                "hideLegend": False,
                #"drillthrough": [],
                #"seriesColumns": None,
                #"crossFilter": [],
                #"crossFilterDisabled": false,
                #"drillthroughDisabled": false,
                #"labelDisabled": false,
            }
        }

    @staticmethod
    def render_barchart(
        query_result: List[Dict[str, Any]],
        title: str,
        description: str,
        chart_type: ChartType = ChartType.DEFAULT,
        x_column: str = None,
        y_columns: List[str] = None,
        series_columns: List[str] = None,
        x_axis_label: str = None,
        y_axis_label: str = None,
    ) -> Dict[str, Any]:
        visual_options = {
            "multipleYAxes": {
                "base": {
                    "id": "-1",
                    "label": y_axis_label,
                    "columns": [],
                    "yAxisScale": "linear",
                    "yAxisMaximumValue": None,
                    "yAxisMinimumValue": None,
                    "horizontalLines": []
                },
                "additional": [],
                "showMultiplePanels": False
            },
            "hideLegend": False,
            "legendLocation": "bottom",
            "xColumnTitle": x_axis_label,
            "xColumn": x_column,
            "yColumns": y_columns,
            "seriesColumns": series_columns,
            "xAxisScale": "linear",
            "verticalLine": "",
            "crossFilterDisabled": False,
            "drillthroughDisabled": False,
            "crossFilter": [],
            "drillthrough": [],
            "selectedDataOnLoad": {
                "all": True,
                "limit": 10
            }
        }

        return {
            "visualType": f"{chart_type.value}bar" if chart_type.value else "bar",
            "title": title,
            "description": description,
            "visualOptions": visual_options
        }

    @staticmethod
    def render_linechart(
        query_result: List[Dict[str, Any]],
        title: str,
        description: str,
        x_column: str = None,
        y_columns: List[str] = None,
        series_columns: List[str] = None,
        x_axis_label: str = None,
        y_axis_label: str = None,
    ) -> Dict[str, Any]:
        visual_options = {
            "multipleYAxes": {
                "base": {
                    "id": "-1",
                    "label": y_axis_label,
                    "columns": [],
                    "yAxisScale": "linear",
                    "yAxisMaximumValue": None,
                    "yAxisMinimumValue": None,
                    "horizontalLines": []
                },
                "additional": [],
                "showMultiplePanels": False
            },
            "hideLegend": False,
            "legendLocation": "bottom",
            "xColumnTitle": x_axis_label,
            "xColumn": x_column,
            "yColumns": y_columns,
            "seriesColumns": series_columns,
            "xAxisScale": "linear",
            "verticalLine": "",
            "crossFilterDisabled": False,
            "drillthroughDisabled": False,
            "crossFilter": [],
            "drillthrough": [],
            "selectedDataOnLoad": {
                "all": True,
                "limit": 10
            }
        }

        return {
            "visualType": "line",
            "title": title,
            "description": description,
            "visualOptions": visual_options
        }

    @staticmethod
    def render_columnchart(
        query_result: List[Dict[str, Any]],
        title: str,
        description: str,
        chart_type: ChartType = ChartType.DEFAULT,
        x_column: str = None,
        y_columns: List[str] = None,
        series_columns: List[str] = None,
        x_axis_label: str = None,
        y_axis_label: str = None,
    ) -> Dict[str, Any]:
        visual_options = {
            "multipleYAxes": {
                "base": {
                    "id": "-1",
                    "label": y_axis_label,
                    "columns": [],
                    "yAxisScale": "linear",
                    "yAxisMaximumValue": None,
                    "yAxisMinimumValue": None,
                    "horizontalLines": []
                },
                "additional": [],
                "showMultiplePanels": False
            },
            "hideLegend": False,
            "legendLocation": "bottom",
            "xColumnTitle": x_axis_label,
            "xColumn": x_column,
            "yColumns": y_columns,
            "seriesColumns": series_columns,
            "xAxisScale": "linear",
            "verticalLine": "",
            "crossFilterDisabled": False,
            "drillthroughDisabled": False,
            "crossFilter": [],
            "drillthrough": [],
            "selectedDataOnLoad": {
                "all": True,
                "limit": 10
            }
        }

        return {
            "visualType": f"{chart_type.value}column" if chart_type.value else "column",
            "title": title,
            "description": description,
            "visualOptions": visual_options
        }

    @staticmethod
    def render_card(
        query_result: List[Dict[str, Any]],
        title: str,
        description: str,
        value_column: str = None,
    ) -> Dict[str, Any]:
        return {
            "visualType": "card",
            "title": title,
            "description": description,
            "visualOptions": {
                "multiStat__textSize": "auto",
                "multiStat__valueColumn": value_column,
                "colorRulesDisabled": False,
                "colorRules": []
            }
        }

    @staticmethod
    def render_multistat(
        query_result: List[Dict[str, Any]],
        title: str,
        description: str,
        value_column: str = None,
        label_column: str = None,
        display_orientation: DisplayOrientation = DisplayOrientation.HORIZONTAL
    ) -> Dict[str, Any]:
        return {
            "visualType": "multistat",
            "title": title,
            "description": description,
            "visualOptions": {
                "multiStat__textSize": "auto",
                "multiStat__valueColumn": value_column,
                "multiStat__labelColumn": label_column,
                "multiStat__displayOrientation": display_orientation.value,
                "multiStat__slot": {"width": 3, "height": 2},
                "colorRulesDisabled": False,
                "colorRules": []
            }
        }

    @staticmethod
    def render_scatterchart(
        query_result: List[Dict[str, Any]],
        title: str,
        description: str,
        x_column: str = None,
        y_columns: List[str] = None,
        series_columns: List[str] = None,
        x_axis_label: str = None,
        y_axis_label: str = None,
    ) -> Dict[str, Any]:
        visual_options = {
            "multipleYAxes": {
                "base": {
                    "id": "-1",
                    "label": y_axis_label,
                    "columns": [],
                    "yAxisScale": "linear",
                    "yAxisMaximumValue": None,
                    "yAxisMinimumValue": None,
                    "horizontalLines": []
                },
                "additional": [],
                "showMultiplePanels": False
            },
            "hideLegend": False,
            "legendLocation": "bottom",
            "xColumnTitle": x_axis_label,
            "xColumn": x_column,
            "yColumns": y_columns,
            "seriesColumns": series_columns,
            "xAxisScale": "linear",
            "verticalLine": "",
            "crossFilterDisabled": False,
            "drillthroughDisabled": False,
            "crossFilter": [],
            "drillthrough": [],
            "selectedDataOnLoad": {
                "all": True,
                "limit": 10
            }
        }

        return {
            "visualType": "scatter",
            "title": title,
            "description": description,
            "visualOptions": visual_options
        }

    @staticmethod
    def render_areachart(
        query_result: List[Dict[str, Any]],
        title: str,
        description: str,
        chart_type: ChartType = ChartType.DEFAULT,
        x_column: str = None,
        y_columns: List[str] = None,
        series_columns: List[str] = None,
        x_axis_label: str = None,
        y_axis_label: str = None,
    ) -> Dict[str, Any]:
        visual_options = {
            "multipleYAxes": {
                "base": {
                    "id": "-1",
                    "label": y_axis_label,
                    "columns": [],
                    "yAxisScale": "linear",
                    "yAxisMaximumValue": None,
                    "yAxisMinimumValue": None,
                    "horizontalLines": []
                },
                "additional": [],
                "showMultiplePanels": False
            },
            "hideLegend": False,
            "legendLocation": "bottom",
            "xColumnTitle": x_axis_label,
            "xColumn": x_column,
            "yColumns": y_columns,
            "seriesColumns": series_columns,
            "xAxisScale": "linear",
            "verticalLine": "",
            "crossFilterDisabled": False,
            "drillthroughDisabled": False,
            "crossFilter": [],
            "drillthrough": [],
            "selectedDataOnLoad": {
                "all": True,
                "limit": 10
            }
        }

        return {
            "visualType": f"{chart_type.value}area" if chart_type.value else "area",
            "title": title,
            "description": description,
            "visualOptions": visual_options
        }

    @staticmethod
    def render_funnelchart(
        query_result: List[Dict[str, Any]],
        title: str,
        description: str,
        x_column: str = None,
        y_column: str = None,
    ) -> Dict[str, Any]:
        visual_options = {
            "xColumn": x_column,
            "yColumns": [y_column],
            "crossFilterDisabled": False,
            "drillthroughDisabled": False,
            "crossFilter": [],
            "drillthrough": []
        }

        return {
            "visualType": "funnel",
            "title": title,
            "description": description,
            "visualOptions": visual_options
        }

    @staticmethod
    def render_mapchart(
        query_result: List[Dict[str, Any]],
        title: str,
        description: str,
        latitude_column: str = None,
        longitude_column: str = None,
    ) -> Dict[str, Any]:
        return {
            "visualType": "map",
            "title": title,
            "description": description,
            "visualOptions": {
                "map__latitudeColumn": latitude_column,
                "map__longitudeColumn": longitude_column
            }
        }

    @staticmethod
    def render_timechart(
        query_result: List[Dict[str, Any]],
        title: str,
        description: str,
        x_datetime_column: str = None,  # Must be datetime column
        y_columns: List[str] = None,
        series_columns: List[str] = None,
        x_axis_label: str = None,
        y_axis_label: str = None,
    ) -> Dict[str, Any]:
        visual_options = {
            "multipleYAxes": {
                "base": {
                    "id": "-1",
                    "label": y_axis_label,
                    "columns": [],
                    "yAxisScale": "linear",
                    "yAxisMaximumValue": None,
                    "yAxisMinimumValue": None,
                    "horizontalLines": []
                },
                "additional": [],
                "showMultiplePanels": False
            },
            "hideLegend": False,
            "legendLocation": "bottom",
            "xColumnTitle": x_axis_label,
            "xColumn": x_datetime_column,
            "yColumns": y_columns,
            "seriesColumns": series_columns,
            "xAxisScale": "linear",
            "verticalLine": "",
            "crossFilterDisabled": False,
            "drillthroughDisabled": False,
            "crossFilter": [],
            "drillthrough": [],
            "selectedDataOnLoad": {
                "all": True,
                "limit": 10
            }
        }

        return {
            "visualType": "timechart",
            "title": title,
            "description": description,
            "visualOptions": visual_options
        }

    @staticmethod
    def render_anomalychart(
        query_result: List[Dict[str, Any]],
        title: str,
        description: str,
        x_column: str = None,
        y_columns: List[str] = None,
        series_columns: List[str] = None,
        x_axis_label: str = None,
        y_axis_label: str = None,
    ) -> Dict[str, Any]:
        visual_options = {
            "multipleYAxes": {
                "base": {
                    "id": "-1",
                    "label": y_axis_label,
                    "columns": [],
                    "yAxisScale": "linear",
                    "yAxisMaximumValue": None,
                    "yAxisMinimumValue": None,
                    "horizontalLines": []
                },
                "additional": [],
                "showMultiplePanels": False
            },
            "hideLegend": False,
            "legendLocation": "bottom",
            "xColumnTitle": x_axis_label,
            "xColumn": x_column,
            "yColumns": y_columns,
            "seriesColumns": series_columns,
            "xAxisScale": "linear",
            "verticalLine": "",
            "crossFilterDisabled": False,
            "drillthroughDisabled": False,
            "crossFilter": [],
            "drillthrough": [],
            "selectedDataOnLoad": {
                "all": True,
                "limit": 10
            }
        }

        return {
            "visualType": "anomaly",
            "title": title,
            "description": description,
            "visualOptions": visual_options
        }

    @staticmethod
    def render_heatmap(
        query_result: List[Dict[str, Any]],
        title: str,
        description: str,
        x_column: str = None,
        y_column: str = None,
        data_column: str = None,
        color_palette: HeatmapColorPalette = HeatmapColorPalette.BLUE,
        x_axis_label: str = None,
        y_axis_label: str = None,
    ) -> Dict[str, Any]:
        visual_options = {
            "xColumn": x_column,
            "yColumn": y_column,
            "heatMap__dataColumn": data_column,
            "heatMap__colorPaletteKey": color_palette.value,
            "xColumnTitle": x_axis_label,
            "yColumnTitle": y_axis_label,
            "crossFilterDisabled": False,
            "drillthroughDisabled": False,
            "crossFilter": [],
            "drillthrough": []
        }

        return {
            "visualType": "heatmap",
            "title": title,
            "description": description,
            "visualOptions": visual_options
        }
